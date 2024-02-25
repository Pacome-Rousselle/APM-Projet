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
solve_cuda(wfc_blocks_ptr blocks, uint64_t seed, masks *my_masks)
{
    ////printf("\n");
    printf("ON SOLVER CUDA\n");
    printf("entry states \n"); 
    grd_print(NULL, blocks); 
    /*first I'll do the sync version*/
    /*then we'lls ee the async*/
    // const uint64_t seed = blocks->states[0]; //with new version we are getting this from outside ??
    uint64_t iteration = 0;
    

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

    uint32_t col_idx;
    uint32_t row_idx;
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

    /*allocate memory on the device, redict the device pointers */
    masks *dev_masks;
    checkCudaErrors(cudaMalloc((void **)&dev_masks, sizeof(masks)));
    uint64_t *column_masks;
    checkCudaErrors(cudaMalloc((void **)&column_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side)); //column masks will be allocated on the host
    //but will hold the adress returning from the device
    checkCudaErrors(cudaMemcpy(&(dev_masks->column_masks), &column_masks, sizeof(uint64_t *), cudaMemcpyHostToDevice));
    //so  here when we make host to device, we are sending the memory location obtained from the device to the device again so that it will know the adress of it
    uint64_t *row_masks;
    checkCudaErrors(cudaMalloc((void **)&row_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side));
    checkCudaErrors(cudaMemcpy(&(dev_masks->row_masks), &row_masks, sizeof(uint64_t *), cudaMemcpyHostToDevice));
    uint64_t *block_masks;
    checkCudaErrors(cudaMalloc((void **)&block_masks, sizeof(uint64_t) * blocks->grid_side * blocks->grid_side));
    checkCudaErrors(cudaMemcpy(&(dev_masks->block_masks), &block_masks, sizeof(uint64_t *), cudaMemcpyHostToDevice));
    uint64_t safe_exit = 0;
    checkCudaErrors(cudaMemcpy(&(dev_masks->safe_exit), &safe_exit, sizeof(uint64_t), cudaMemcpyHostToDevice));

    uint64_t *host_column_masks = (uint64_t *)malloc(sizeof(uint64_t) * blocks->block_side * blocks->grid_side);
    uint64_t *host_row_masks    = (uint64_t *)malloc(sizeof(uint64_t) * blocks->block_side * blocks->grid_side);
    uint64_t *host_block_masks  = (uint64_t *)malloc(sizeof(uint64_t) * blocks->grid_side * blocks->grid_side);

    int *dev_pendings;
    int *host_pendings = (int*)malloc(sizeof(int) * blocks->block_side * blocks->block_side * blocks->grid_side * blocks->grid_side);
    memset(host_pendings, -1, sizeof(int) * blocks->block_side * blocks->block_side * blocks->grid_side * blocks->grid_side);
    checkCudaErrors(cudaMalloc((void **)&dev_pendings, sizeof(int) * blocks->block_side * blocks->block_side * blocks->grid_side * blocks->grid_side));

    int *gpa_return_val;
    int host_gpa_return_val = 5;
    int *dev_next;

int host_next; 

uint64_t* dev_conf_loc; 
uint64_t* dev_pass_collapsed; 

uint64_t* dev_place; 
checkCudaErrors(cudaMalloc((void **)&dev_place, sizeof(uint64_t)));
checkCudaErrors(cudaMalloc((void **)&dev_pass_collapsed, sizeof(uint64_t)));

checkCudaErrors(cudaMalloc((void **)&dev_conf_loc, sizeof(uint64_t)));
uint64_t host_conf_loc; 
uint64_t host_place; 
uint64_t host_pass_collapsed; 


uint64_t* dev_th_loc; 
checkCudaErrors(cudaMalloc((void **)&dev_th_loc, sizeof(uint64_t)));
uint64_t host_th_loc; 

    checkCudaErrors(cudaMalloc((void **)&gpa_return_val, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&dev_next, sizeof(int)));

    forever {
        checkCudaErrors(cudaMemcpy(dev_pendings, host_pendings, sizeof(int) * blocks->block_side * blocks->block_side * blocks->grid_side * blocks->grid_side, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(dev_next, 0, sizeof(int)));
        //make the execution
        //since all the threads attacks a block in the grid and they are returning their piece, we dont have to pass
   printf("before dev entropy\n"); 
   grd_print(NULL, h_blocks);

           for (int i = 0; i < nb_threads; i++) {
        dev_entropy_location_per_thread[i].entropy          = UINT8_MAX;
        dev_entropy_location_per_thread[i].location_in_blk  = { 0, 0 };
        dev_entropy_location_per_thread[i].location_in_grid = { 0, 0 };
        dev_entropy_location_per_thread[i]._1               = 0;
        dev_entropy_location_per_thread[i]._2               = 0;
    }
    checkCudaErrors(cudaMemcpy(dev_ret_entropy_location_per_thread, dev_entropy_location_per_thread, sizeof(entropy_location) * nb_threads, cudaMemcpyHostToDevice));

        //grid information to them => it is known internally
       /* dev_blk_min_entropy<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks, dev_ret_entropy_location_per_thread);
        //return the array from device to host
        checkCudaErrors(cudaMemcpy(dev_entropy_location_per_thread,
                                   dev_ret_entropy_location_per_thread, sizeof(entropy_location) * nb_threads, cudaMemcpyDeviceToHost));
        //cudaDeviceSynchronize();
loc.entropy == UINT8_MAX; 

        //finding the minimum problem
        int keep; 
        for (int i = 0; i < nb_threads; i++) {
            test = dev_entropy_location_per_thread[i];
            if (test.entropy < loc.entropy)
                {loc = test;
                keep = i; 
                }
        }
        printf("keep: %d", keep); 
        if (loc.entropy == UINT8_MAX)
            break;

for(int i = 0; i < nb_threads; i++){
    printf("i %d\n", i ); 
    printf(" gx %lu, gy %lu,  x %lu, y %lu \n", 
                                        dev_entropy_location_per_thread[i].location_in_grid.x,
                                        dev_entropy_location_per_thread[i].location_in_grid.y,
                                        dev_entropy_location_per_thread[i].location_in_blk.x,
                                        dev_entropy_location_per_thread[i].location_in_blk.y); 
}

        loc = dev_entropy_location_per_thread[keep]; 
        */
        loc.entropy = UINT8_MAX;
        for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++)
            {
                test = blk_min_entropy(h_blocks,gx,gy);
                if(test.entropy < loc.entropy)
                    loc = test;
            }



        int h_idx = get_thread_glob_idx(h_blocks,
                                        loc.location_in_grid.x,
                                        loc.location_in_grid.y,
                                        loc.location_in_blk.x,
                                        loc.location_in_blk.y);
        //printf("index to propagate %d \n ", h_idx);
        printf(" index to propagate %d, gx %lu, gy %lu,  x %lu, y %lu \n", h_idx,
                                        loc.location_in_grid.x,
                                        loc.location_in_grid.y,
                                        loc.location_in_blk.x,
                                        loc.location_in_blk.y); 

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
        col_idx   = loc.location_in_grid.y * blocks->block_side + loc.location_in_blk.y; //between 0 to 8
        row_idx   = loc.location_in_grid.x * blocks->block_side + loc.location_in_blk.x; //between 0 to 8
        block_idx = loc.location_in_grid.x * blocks->grid_side  + loc.location_in_grid.y; //between 0 to 8
        printf("col_ids %d, row odx %d,  blockidx %d", col_idx, row_idx, block_idx); 

        printf("after collapsing\n");
        grd_print(NULL, h_blocks);
        //try to import this to the device later
        
        set_mask(my_masks, col_idx, row_idx, block_idx, h_blocks->states[h_idx]);
         print_masks(my_masks, blocks->block_side, blocks->grid_side);

        if (my_masks->safe_exit == 1) {
            printf("conflict while collapsing, safe exiting in solver cuda \n");
            return false;
        }
        checkCudaErrors(cudaMemcpy(column_masks, my_masks->column_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(row_masks, my_masks->row_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(block_masks, my_masks->block_masks, sizeof(uint64_t) * blocks->grid_side * blocks->grid_side, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&(dev_masks->safe_exit), &my_masks->safe_exit, sizeof(uint64_t), cudaMemcpyHostToDevice));


        checkCudaErrors(cudaMemcpy(d_blocks, h_blocks, size, cudaMemcpyHostToDevice)); //get this outside of forever ?

        //CALL TO PROPOGATE ALL
      

        global_propagate_all<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks, loc.location_in_grid.x,
                                                                loc.location_in_grid.y,
                                                                loc.location_in_blk.x,
                                                                loc.location_in_blk.y, h_blocks->states[h_idx], dev_masks,
                                                                dev_pendings, gpa_return_val, dev_next, dev_conf_loc, dev_th_loc, dev_place, 
                                                                dev_pass_collapsed);


        checkCudaErrors(cudaMemcpy(&host_gpa_return_val, gpa_return_val, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&host_conf_loc, dev_conf_loc, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&host_th_loc, dev_th_loc, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&host_place, dev_place, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&host_pass_collapsed, dev_pass_collapsed, sizeof(uint64_t), cudaMemcpyDeviceToHost));


        //printf("return val: %d\n", host_gpa_return_val);

        if (host_gpa_return_val == 1) {
            printf("encountered an error or a conflict safe exiting dev_confloc %lu , dev_th_loc %lu\n", host_conf_loc, host_th_loc);
            return false;
        }

        checkCudaErrors(cudaMemcpy(h_blocks, d_blocks, size, cudaMemcpyDeviceToHost));
         printf("GPU AFTER propogate all\n");
         grd_print(NULL, h_blocks);
        cudaMemcpy(my_masks, dev_masks, sizeof(masks *), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaMemcpy(host_column_masks, column_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(host_row_masks, row_masks, sizeof(uint64_t) * blocks->block_side * blocks->grid_side, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(host_block_masks, block_masks, sizeof(uint64_t) * blocks->grid_side * blocks->grid_side, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&safe_exit, &(dev_masks->safe_exit), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        my_masks->column_masks = host_column_masks;
        my_masks->row_masks    = host_row_masks;
        my_masks->block_masks  = host_block_masks;
        my_masks->safe_exit    = safe_exit;
         print_masks(my_masks, blocks->block_side, blocks->grid_side);

          if (my_masks->safe_exit == 1) {
            printf("conflict while collapsing,received from the solver \n");
            printf("encountered an error or a conflict safe exiting dev_confloc %lu , dev_th_loc %lu, place %d, coll %d\n",
                         host_conf_loc, host_th_loc, host_place, bit_to_val(host_pass_collapsed));

            return false;
        }

        //checkCudaErrors(cudaMemcpy(host_pendings, dev_pendings, sizeof(int) * blocks->block_side * blocks->block_side * blocks->grid_side * blocks->grid_side, cudaMemcpyDeviceToHost));

//for(int i = 0; i < blocks->block_side * blocks->block_side * blocks->grid_side * blocks->grid_side; i++ )
//{
//    printf("pendings[%d] = %d \n", i, host_pendings[i]); 
//}
//printf("nb iteration %d", iteration); 
//        checkCudaErrors(cudaMemcpy(&host_next, dev_next, sizeof(int) , cudaMemcpyDeviceToHost));
//        printf("host next %d\n", host_next); 

        iteration += 1;
        //if (iteration == 5) {
        //    ////printf("iteration = 1 \n");
        //    break;
        //}

        //free the memory locations from the cpu and gpu!!!
        //if it's outside of forever call teh destroyer
    }

    return false;
};
