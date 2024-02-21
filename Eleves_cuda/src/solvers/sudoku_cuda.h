bool
verify_states(wfc_blocks_ptr blocks_ref, wfc_blocks_ptr blocks_test)
{
    int idx;
    /*check if the meta data is the same */
    if ((blocks_ref->grid_side != blocks_test->grid_side) ||
        (blocks_ref->block_side != blocks_test->block_side)) {
        printf("gridor block sizes are not the same\n");
        return false;
    }

    /*check if the states are the same */
    for (size_t gx = 0; gx < blocks_ref->grid_side; gx++) {
        for (size_t gy = 0; gy < blocks_ref->grid_side; gy++) {
            for (size_t x = 0; x < blocks_ref->block_side; x++) {
                for (size_t y = 0; y < blocks_ref->block_side; y++) {
                    idx = get_thread_glob_idx(blocks_ref, gx, gy, x, y);
                    if (blocks_ref->states[idx] != blocks_test->states[idx]) {
                        printf("states are not the same\n");
                        printf("location ref: gx %lu gy %lu x %lu y %lu\n",
                               gx, gy, x, y);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

void
grd_print_numbers(FILE *const file, const wfc_blocks_ptr block)
{
    FILE *fp = file;
    if (fp == NULL)
        fp = stdout;

    uint8_t gs = block->grid_side;
    uint8_t bs = block->block_side;

    for (uint32_t ii = 0; ii < gs; ii++) {
        for (uint32_t i = 0; i < gs; i++) {
            fprintf(fp, "+");
            for (uint32_t j = 0; j < bs; j++) {
                fprintf(fp, "----------+");
            }
            fprintf(fp, "   ");
        }
        fprintf(fp, "\n");

        for (uint32_t jj = 0; jj < bs; jj++) {
            for (uint32_t i = 0; i < gs; i++) {
                fprintf(fp, "|");
                for (uint32_t j = 0; j < bs; j++) {
                    const uint64_t collapsed = *blk_at(block, ii, i, jj, j);
                    printf("  %lu  ", collapsed);
                    fprintf(fp, " |");
                }
                fprintf(fp, "   ");
            }
            fprintf(fp, "\n");

            for (int i = 0; i < gs; i++) {
                fprintf(fp, "+");
                for (int j = 0; j < bs; j++) {
                    fprintf(fp, "----------+");
                }
                fprintf(fp, "   ");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }
}
__global__ void
show_index(wfc_blocks_ptr d_blocks)
{//for visualizing purposes I am leaving my comments an explaining everything step by step
    //one thread is getting a block 
    //therefore it should traverse each element of the block 

    
    //here each thread gets an element of a block 

    /*
    int block_size = d_blocks->block_side;
    int grid_size  = d_blocks->grid_side;

    int row        = blockIdx.y * blockDim.y + threadIdx.y; // just changed x and y here
    int col        = blockIdx.x * blockDim.x + threadIdx.x; // just changed x and y here
    int width      = gridDim.x * blockDim.x;                 // M = a* N

    
    d_blocks->states[(row  * width + col)] = row  * width + col; //row*width+col;
    */
   
//==> so each thread should leave behind some elemens 
//here we see that each thread got the first element of a grid
    /*
    int block_size = d_blocks->block_side;
    int grid_size  = d_blocks->grid_side;

    int row        = blockIdx.y * blockDim.y + threadIdx.y; // just changed x and y here
    int col        = blockIdx.x * blockDim.x + threadIdx.x; // just changed x and y here
    int width      = gridDim.x * blockDim.x;                 // M = a* N

    
    d_blocks->states[(row  * width + col)*block_size*block_size] = row  * width + col; //row*width+col;
*/
   //now each thread takes care of a block 
    int block_size = d_blocks->block_side;
    int grid_size  = d_blocks->grid_side;

    int row        = blockIdx.y * blockDim.y + threadIdx.y; // just changed x and y here
    int col        = blockIdx.x * blockDim.x + threadIdx.x; // just changed x and y here
    int width      = gridDim.x * blockDim.x;                 // M = a* N

    for(int i = 0; i < block_size*block_size; i++)
    d_blocks->states[(row  * width + col)*block_size*block_size +i ] = (row  * width + col)*block_size*block_size +i; //row*width+col;

}

void see_normal_indexes(wfc_blocks_ptr blocks_ref)
{
int idx;

   for (size_t gx = 0; gx < blocks_ref->grid_side; gx++) {
        for (size_t gy = 0; gy < blocks_ref->grid_side; gy++) {
            for (size_t x = 0; x < blocks_ref->block_side; x++) {
                for (size_t y = 0; y < blocks_ref->block_side; y++) {
                    idx = get_thread_glob_idx(blocks_ref, gx, gy, x, y);
                    
                    blocks_ref->states[idx] = idx; 
                }
            }
        }
    }
}

__global__ void dev_get_thread_glob_idx(){

}

__device__ void 
dev_bitfield_count(uint64_t x, uint8_t* ret)
{
    const uint64_t m1  = 0x5555555555555555; //binary: 0101...
    const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
    const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...

    x -= (x >> 1) & m1;                //put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2);    //put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & m4;           //put count of each 8 bits into those 8 bits
    (*ret)=  (uint8_t)((x * h01) >> 56); //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

__device__ void
dev_entropy_compute(uint64_t state,  uint8_t* ret)
{
    dev_bitfield_count(state, ret);
}

__global__ void 
dev_blk_min_entropy(const wfc_blocks_ptr blocks, entropy_location* entropy_loctaion_per_thread)
{//each thread will enter here and will find the location for its own block, 
//will then stock it in the dedicated case for themselves in the entropy_location* entropy_loctaion_per_thread
//hmm but it means that the new_loc array then will cointain pointers from gpu    
    vec2 blk_location   = { 0 };
    uint8_t min_entropy = UINT8_MAX;
    uint8_t entropy_test;
    int idx;

    int block_size = blocks->block_side;
    int grid_size  = blocks->grid_side;

    int row        = blockIdx.y * blockDim.y + threadIdx.y; // just changed x and y here
    int col        = blockIdx.x * blockDim.x + threadIdx.x; // just changed x and y here
    int width      = gridDim.x * blockDim.x;                 // M = a* N
   
    //this gives the number of the case that thread starts to read from 
    int th_idx_arr_location  = (row  * width + col)*block_size*block_size; //therefore it is not thread idx really 
    int th_idx = row * width + col; 
    
    //Navigate through the block
    for (int block_x = 0; block_x < blocks->block_side; block_x++)
        {
        for (int block_y = 0; block_y < blocks->block_side; block_y++) {
             idx          =  th_idx_arr_location + block_x  + block_y;
             dev_entropy_compute(blocks->states[idx], &entropy_test);
            if ((entropy_test < min_entropy) && (entropy_test > 1)) {
                min_entropy    = entropy_test;
                blk_location.x = block_x;
                blk_location.y = block_y;
            }
        }
        }
    
    int our_grid_Idx = blockIdx.x / grid_size; 
    int our_grid_Idy = blockIdx.x % grid_size;

    entropy_loctaion_per_thread[th_idx].entropy            = min_entropy;
    entropy_loctaion_per_thread[th_idx].location_in_blk    = blk_location;
    entropy_loctaion_per_thread[th_idx].location_in_grid.x = our_grid_Idx;  
    entropy_loctaion_per_thread[th_idx].location_in_grid.y = our_grid_Idy; 



    //return new;
}