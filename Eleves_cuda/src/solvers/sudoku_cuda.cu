//#define _GNU_SOURCE

/*I am trying to add excessive but useful comments
So that we both remember when we read, and write a good report 
for example I think we may visualize how threads are spreat 
A serious question: is it a good idea that each thread takes a block ? 
Another: where can we use streams ? Would it be an overkill considering the problem size ? 
Async => is the operation order of &=~ on elements is important ? 
==>> it must be NOT  because in the subject description it says that : 
sudoku is a carree latin :nah it seems like there is no relation but i am not sure ?  

*/

#include "bitfield.h"
#include "wfc.h"

#include "../inc/helper_cuda.h"
#include "sudoku_cuda.h" //functions to be called when inside gpu are defined here

bool
solve_cuda(wfc_blocks_ptr blocks, uint64_t seed, masks* my_masks)
{
    ////printf("\n");
    printf("ON SOLVER CUDA\n");
    /*first I'll do the sync version*/
    /*then we'lls ee the async*/
   // const uint64_t seed = blocks->states[0]; //with new version we are getting this from outside ??
    uint64_t iteration  = 0;
    // Set seed
    //const uint64_t seed = blocks->states[0];
    ////printf("SEED%lu\n", seed);
    //struct {
    //    uint32_t gy, x, y, _1;
    //    uint64_t state;
    //} row_changes[blocks->grid_side];

    entropy_location loc, test;
    loc.entropy = UINT8_MAX;
    int changed = 0;

    //while find_min_entropy in grid
    //read dev_blk_min_entropy first then read here:
    //so here we have to pass a return array
    //=> after gpu execution we are going to have an array of entropy location
    //so the problem below becomes a problem of
    //finding the minimum entropy within the array
    //and keeping the pointer to the structure
    /*loc.entropy = UINT8_MAX;hop
         for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++)
            {
                test = blk_min_entropy(blocks,gx,gy);
                if(test.entropy < loc.entropy)
                    loc = test;
            }
            */

uint32_t col_idx  ;
uint32_t row_idx  ;
uint32_t block_idx;
    //create an array of entropy location => gonna do that for each iteration so dont forgetto free
    //to be or not to be a malloc host ?

    int grid_size           = blocks->grid_side;
    int block_size          = blocks->block_side;
    int nb_threads          = grid_size * grid_size;
    size_t thread_per_block = 1; //each thread will get one wfc block
    size_t nb_blocks        = grid_size * grid_size;
    dim3 dim_cuda_block(thread_per_block, 1, 1);     //1 dimensional blocs
    dim3 dim_cuda_grid(grid_size * grid_size, 1, 1); //1 dimensional grid

    const uint64_t size = (wfc_control_states_count(grid_size, block_size) * sizeof(uint64_t)) +
                          (grid_size * grid_size * block_size * block_size * sizeof(uint64_t)) +
                          sizeof(wfc_blocks);


        wfc_blocks_ptr h_blocks         = NULL;
        wfc_blocks_ptr h_blocks_changed = NULL;

        //allocate the data space on host and pin it
        checkCudaErrors(cudaMallocHost((void **)&h_blocks, size));
        //initialize the pinned data space
        memcpy(h_blocks, blocks, size); 
        //
        wfc_blocks_ptr d_blocks;
        //allocate space on the device
        checkCudaErrors(cudaMalloc((void **)&d_blocks, size)); //mempitch => is it padded? 
        //copy the pinned data from the host to device
        checkCudaErrors(cudaMemcpy(d_blocks, h_blocks, size, cudaMemcpyHostToDevice)); 
         entropy_location *dev_entropy_location_per_thread = NULL;
        //it will stock the return fromt the device
        entropy_location *dev_ret_entropy_location_per_thread = NULL;
        dev_ret_entropy_location_per_thread                   = NULL; //(entropy_location *)malloc(sizeof(entropy_location) * nb_threads);

        dev_entropy_location_per_thread = (entropy_location *)malloc(sizeof(entropy_location) * nb_threads);
    
        //initialize it =>making sure
        for (int i = 0; i < nb_threads; i++) {
            dev_entropy_location_per_thread[i].entropy          = UINT8_MAX;
            dev_entropy_location_per_thread[i].location_in_blk  = { 0, 0 };
            dev_entropy_location_per_thread[i].location_in_grid = { 0, 0 };
            dev_entropy_location_per_thread[i]._1               = 0;
            dev_entropy_location_per_thread[i]._2               = 0;
        }

         //allocate space in the gpu
        checkCudaErrors(cudaMalloc((void **)&dev_ret_entropy_location_per_thread, sizeof(entropy_location) * nb_threads)); //get this outside of forever too ? because it will be overwritten each time
         //initialize the allocated space in the gpu
        checkCudaErrors(cudaMemcpy(dev_ret_entropy_location_per_thread, dev_entropy_location_per_thread, sizeof(entropy_location) * nb_threads, cudaMemcpyHostToDevice));
        
masks* dev_masks;
cudaMalloc((void**)&dev_masks, sizeof(masks));

uint64_t* column_masks;
cudaMalloc((void**)&column_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side); //column masks will be allocated on the host 
//but will hold the adress returning from the device 
cudaMemcpy(&(dev_masks->column_masks), &column_masks, sizeof(uint64_t*), cudaMemcpyHostToDevice);
//so  here when we make host to device, we are sending the memory location obtained from the device to the device again so that it will know the adress of it 
uint64_t* row_masks;
cudaMalloc((void**)&row_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side);
cudaMemcpy(&(dev_masks->row_masks), &row_masks, sizeof(uint64_t*), cudaMemcpyHostToDevice);
uint64_t* block_masks;
cudaMalloc((void**)&block_masks, sizeof(uint64_t) * blocks->grid_side * blocks->grid_side);
cudaMemcpy(&(dev_masks->block_masks), &block_masks, sizeof(uint64_t*), cudaMemcpyHostToDevice);
uint64_t safe_exit = 0;
cudaMemcpy(&(dev_masks->safe_exit), &safe_exit, sizeof(uint64_t), cudaMemcpyHostToDevice);

uint64_t* host_column_masks = (uint64_t*)malloc(sizeof(uint64_t) * blocks->block_side * blocks->grid_side);
uint64_t* host_row_masks = (uint64_t*)malloc(sizeof(uint64_t) * blocks->block_side * blocks->grid_side);
uint64_t* host_block_masks = (uint64_t*)malloc(sizeof(uint64_t) * blocks->grid_side* blocks->grid_side);

    forever {
        

        //make the execution
        //since all the threads attacks a block in the grid and they are returning their piece, we dont have to pass
        //grid information to them => it is known internally
        dev_blk_min_entropy<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks, dev_ret_entropy_location_per_thread);
        //return the array from device to host
        checkCudaErrors(cudaMemcpy(dev_entropy_location_per_thread,
                                   dev_ret_entropy_location_per_thread, sizeof(entropy_location) * nb_threads, cudaMemcpyDeviceToHost));
        //cudaDeviceSynchronize();

        ////printf("making sure they are different ? \n");
        //for (int i = 0; i < nb_threads; i++) {
        //    printf("%" PRIu8 "\n %d %d %d %d \n", dev_entropy_location_per_thread[i].entropy,
        //           dev_entropy_location_per_thread[i].location_in_blk.x,
        //           dev_entropy_location_per_thread[i].location_in_blk.y,
        //           dev_entropy_location_per_thread[i].location_in_grid.x,
        //           dev_entropy_location_per_thread[i].location_in_grid.y);
        //}

        //finding the minimum problem
        for (int i = 0; i < nb_threads; i++) {
            test = dev_entropy_location_per_thread[i];
            if (test.entropy < loc.entropy)
                loc = test;
        }
        if (loc.entropy == UINT8_MAX)
            break;        


        ////grd_print(NULL, blocks); //last grid encountered
        //printf("printing the entropy array minimum: \n");
        //printf("%" PRIu8 "\n %d %d %d %d \n", loc.entropy,
        //       loc.location_in_blk.x,
        //       loc.location_in_blk.y,
        //       loc.location_in_grid.x,
        //       loc.location_in_grid.y);
        //calling the same function from the cpu solver to compare the results
        //i will not delete them until lost moment! if we have time i will do the streams too and dont want to write the same thing again!

        int h_idx = get_thread_glob_idx(h_blocks,
                                      loc.location_in_grid.x,
                                      loc.location_in_grid.y,
                                      loc.location_in_blk.x,
                                      loc.location_in_blk.y);
        //printf("index to propagate %d \n ", h_idx); 

        //it will be called from the cpu
        //since there is only one state to pass i dont see it necessary to do inside gpu
        h_blocks->states[h_idx] = entropy_collapse_state(h_blocks->states[h_idx],
                                                     loc.location_in_grid.x,
                                                     loc.location_in_grid.y,
                                                     loc.location_in_blk.x,
                                                     loc.location_in_blk.y,
                                                     seed,
                                                     iteration);
                                          //ABOVE: change in one state
        col_idx   = loc.location_in_grid.y * blocks->block_side + loc.location_in_blk.y; 
        row_idx   = loc.location_in_grid.x * blocks->block_side + loc.location_in_blk.x; 
        block_idx = loc.location_in_grid.x * blocks->grid_side  + loc.location_in_grid.y; 
        printf("after collapsing\n"); 
        grd_print(NULL, blocks); 
        //try to import this to the device later 
        set_mask(my_masks, col_idx, row_idx, block_idx, h_blocks->states[h_idx]); 
        printf("masks on cpu\n"); 
        print_masks(my_masks, blocks -> block_side, blocks-> grid_side); 
//but will hold the adress returning from the device 
//checkCudaErrors(cudaMemcpy(dev_masks, my_masks, sizeof(masks*), cudaMemcpyHostToDevice));
checkCudaErrors (cudaMemcpy( column_masks, my_masks->column_masks,sizeof(uint64_t)* blocks->block_side * blocks->grid_side, cudaMemcpyHostToDevice));
checkCudaErrors (cudaMemcpy( row_masks,    my_masks->row_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side, cudaMemcpyHostToDevice));
checkCudaErrors (cudaMemcpy( block_masks,  my_masks->block_masks, sizeof(uint64_t) * blocks->grid_side * blocks->grid_side, cudaMemcpyHostToDevice));
checkCudaErrors (cudaMemcpy( &(safe_exit), &my_masks->safe_exit, sizeof(uint64_t), cudaMemcpyHostToDevice));

        printf("masks on gpu\n");


        //testing will take time so i continue for now
        //printf(" blocks->states[idx] GPU: %lu\n", h_blocks->states[h_idx]);

        if(my_masks->safe_exit == 1)
        {
            printf("conflict while collapsing, safe exiting\n"); 
            return false; 
        }


        checkCudaErrors(cudaMemcpy(d_blocks, h_blocks, size, cudaMemcpyHostToDevice)); //get this outside of forever ?
        

        //block propagate
                                                                                    //here gotta pass the states from h_blocks, otherwise it wont recognize; it is expecting the value from cpu 
        /*
        dev_blk_propagate<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks, loc.location_in_grid.x,
                                                             loc.location_in_grid.y, h_blocks->states[h_idx]);
        
        
        //just to print, transferring data 
        //after verification remove it 
        //checkCudaErrors(cudaMemcpy(h_blocks,d_blocks, size,cudaMemcpyDeviceToHost));
        ////printf("GPU AFTER BLOCK PROPAGATE\n"); 
        ////grd_print(NULL, h_blocks); 

        dev_grd_propagate_column<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks, loc.location_in_grid.x,
           loc.location_in_grid.y,  loc.location_in_blk.x, loc.location_in_blk.y,h_blocks->states[h_idx]);
        //checkCudaErrors(cudaMemcpy(h_blocks,d_blocks, size,cudaMemcpyDeviceToHost));
        ////printf("GPU AFTER COLUMN PROPAGATE\n"); 
        ////grd_print(NULL, h_blocks); 


        dev_grd_propagate_row<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks, loc.location_in_grid.x,
            loc.location_in_grid.y, loc.location_in_blk.x, loc.location_in_blk.y,h_blocks->states[h_idx]);
       */

        checkCudaErrors(cudaMemcpy(h_blocks,d_blocks, size,cudaMemcpyDeviceToHost));
        printf("GPU AFTER ROW PROPAGATE\n"); 
        grd_print(NULL, h_blocks); 
//cudaMemcpy(my_masks, dev_masks, sizeof(masks*), cudaMemcpyDeviceToHost);
checkCudaErrors(cudaMemcpy( host_column_masks, column_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy( host_row_masks,      row_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy( host_block_masks,    block_masks, sizeof(uint64_t) * blocks->grid_side * blocks->grid_side, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy( &my_masks->safe_exit, &(dev_masks->safe_exit), sizeof(uint64_t), cudaMemcpyDeviceToHost)) ;
my_masks->column_masks = column_masks; 
my_masks->row_masks = row_masks; 
my_masks->block_masks = block_masks; 
print_masks(my_masks, blocks->block_side, blocks->grid_side); 
/// ERROR CHECK ///

        //int* dev_result = NULL; //we need the return frm the function that is executing in 
        ////the gpu, therefore to return a value we have to allocate on the device 
        //int result = 0 ; 
        //checkCudaErrors(cudaMalloc((void**)& dev_result, sizeof(uint8_t))); 
        ////call grid error check 
        //dev_grd_check_error_in_blk<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks,loc.location_in_grid.x,
        //    loc.location_in_grid.y, loc.location_in_blk.x, loc.location_in_blk.y, dev_result); 
        ////pass dev result and retrieve it as the result 
        //checkCudaErrors(cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost)); 
         
        /////////CPU///////////////////
        /*loc.entropy = UINT8_MAX;
        for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++) {
                test = blk_min_entropy(blocks, gx, gy);
                if (test.entropy < loc.entropy)
                    loc = test;
            }
        //////printf("CPU RESULTS\n");
        //////printf("%" PRIu8 "\n %d %d %d %d \n", loc.entropy,
        //       loc.location_in_blk.x,
        //       loc.location_in_blk.y,
        //       loc.location_in_grid.x,
        //       loc.location_in_grid.y);
        int idx = get_thread_glob_idx(h_blocks,
                                      loc.location_in_grid.x,
                                      loc.location_in_grid.y,
                                      loc.location_in_blk.x,
                                      loc.location_in_blk.y);
        blocks->states[idx] = entropy_collapse_state(blocks->states[idx],
                                                     loc.location_in_grid.x,
                                                     loc.location_in_grid.y,
                                                     loc.location_in_blk.x,
                                                     loc.location_in_blk.y,
                                                     seed,
                                                     iteration);
        ////printf("index to propagate %d \n ", idx); 

        ////printf(" blocks->states[idx] CPU: %lu\n", blocks->states[idx]);
//add cpu function here that would help me to verify the steps 
        

        if(verify_states(h_blocks, blocks)){
            ////printf("VERIFIED\n");
        }else {
            ////printf("NOT VERIFIED\n"); 
        }; 
*/
        iteration += 1;
        if (iteration == 1) {
            ////printf("iteration = 1 \n");
            break;
        }

        //free the memory locations from the cpu and gpu!!!
        //if it's outside of forever call teh destroyer
    }

    return false;
};
