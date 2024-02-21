#define _GNU_SOURCE

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
solve_cuda(wfc_blocks_ptr blocks)
{
    printf("\n");
    printf("ON SOLVER CUDA\n");
    /*first I'll do the sync version*/
    /*then we'lls ee the async*/
    uint64_t iteration = 0;
    // Set seed
    //const uint64_t seed = blocks->states[0];
    //printf("SEED%lu\n", seed);
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
    /*loc.entropy = UINT8_MAX;
         for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++)
            {
                test = blk_min_entropy(blocks,gx,gy);
                if(test.entropy < loc.entropy)
                    loc = test;
            }
            */
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

    forever {
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
        checkCudaErrors(cudaMalloc((void**)& dev_ret_entropy_location_per_thread, sizeof(entropy_location)* nb_threads)); 
        //initialize the allocated space in the gpu
        checkCudaErrors(cudaMemcpy(dev_ret_entropy_location_per_thread, dev_entropy_location_per_thread, sizeof(entropy_location) * nb_threads, cudaMemcpyHostToDevice));
        //make the execution
        //since all the threads attacks a block in the grid and they are returning their piece, we dont have to pass
        //grid information to them => it is known internally
        dev_blk_min_entropy<<<dim_cuda_grid, dim_cuda_block>>>(d_blocks, dev_ret_entropy_location_per_thread);
        //return the array from device to host
        checkCudaErrors(cudaMemcpy((void **)dev_entropy_location_per_thread, 
                        dev_ret_entropy_location_per_thread, sizeof(entropy_location) * 
                        nb_threads, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        printf("making sure they are different ? \n"); 
        for (int i = 0; i < nb_threads; i++) {
            printf("%" PRIu8 "\n %d %d %d %d \n", dev_entropy_location_per_thread[i].entropy,
                   dev_entropy_location_per_thread[i].location_in_blk.x,
                   dev_entropy_location_per_thread[i].location_in_blk.y,
                   dev_entropy_location_per_thread[i].location_in_grid.x,
                   dev_entropy_location_per_thread[i].location_in_grid.y);
        }

        //finding the minimum problem
        for (int i = 0; i < nb_threads; i++) {
            test = dev_entropy_location_per_thread[i];
            if (test.entropy < loc.entropy)
                loc = test;
        }
        grd_print(NULL, blocks); 
        printf("printing the entropy array minimum: \n");
        printf("%" PRIu8 "\n %d %d %d %d \n", loc.entropy,
               loc.location_in_blk.x,
               loc.location_in_blk.y,
               loc.location_in_grid.x,
               loc.location_in_grid.y);
        //calling the same function from the cpu solver to compare the results
        //i will not delete them until lost moment! if we have time i will do the streams too and dont want to write the same thing again!
        loc.entropy = UINT8_MAX;
        //it find the minimum entropy for a given grid part, therefore all the things i did above wasnt necessary?! 
        //    => can we not use it somewhere else 
        for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++) {
                test = blk_min_entropy(blocks, gx, gy);
                if (test.entropy < loc.entropy)
                    loc = test;
            }
        printf("CPU RESULTS\n");
        printf("%" PRIu8 "\n %d %d %d %d \n", loc.entropy,
               loc.location_in_blk.x,
               loc.location_in_blk.y,
               loc.location_in_grid.x,
               loc.location_in_grid.y);

        iteration += 1;
        if (iteration == 1) {
            printf("iteration = 1 \n");
            break;
        }
    }

    return false;
};
