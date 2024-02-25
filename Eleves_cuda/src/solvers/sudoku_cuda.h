bool
verify_states(wfc_blocks_ptr blocks_ref, wfc_blocks_ptr blocks_test)
{
    int idx;
    /*check if the meta data is the same */
    if ((blocks_ref->grid_side != blocks_test->grid_side) ||
        (blocks_ref->block_side != blocks_test->block_side)) {
        printf("gridor block sizes are not the same\n");
        printf("blocks_ref->grid_side : %u blocks_test->grid_side: %u", blocks_ref->grid_side, blocks_test->grid_side);
        printf("blocks_ref->block_side : %u  blocks_test->block_side: %u",
               blocks_ref->block_side, blocks_test->block_side);
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
{ //for visualizing purposes I am leaving my comments an explaining everything step by step
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

    int row   = blockIdx.y * blockDim.y + threadIdx.y; // just changed x and y here
    int col   = blockIdx.x * blockDim.x + threadIdx.x; // just changed x and y here
    int width = gridDim.x * blockDim.x;                // M = a* N

    for (int i = 0; i < block_size * block_size; i++)
        d_blocks->states[(row * width + col) * block_size * block_size + i] = (row * width + col) * block_size * block_size + i; //row*width+col;
}

void
see_normal_indexes(wfc_blocks_ptr blocks_ref)
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

__device__ void
dev_get_thread_glob_idx(const wfc_blocks_ptr blocks, uint8_t grid_X, uint8_t grid_Y, uint8_t block_X, uint8_t block_Y, int ret)
{
    int block_id   = grid_X * blocks->grid_side + grid_Y;
    int Nb_threads = blocks->block_side * blocks->block_side;
    int local_id   = block_X * blocks->block_side + block_Y;
    ret            = block_id * Nb_threads + local_id;
}

__device__ void
dev_bitfield_count(uint64_t x, uint8_t *ret)
{
    const uint64_t m1  = 0x5555555555555555; //binary: 0101...
    const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
    const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...

    x -= (x >> 1) & m1;                  //put count of each 2 bits into those 2 bits
    x      = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
    x      = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits
    (*ret) = (uint8_t)((x * h01) >> 56); //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

__device__ void
dev_entropy_compute(uint64_t state, uint8_t *ret)
{
    dev_bitfield_count(state, ret);
}

__global__ void
dev_blk_min_entropy(const wfc_blocks_ptr blocks, entropy_location *entropy_loctaion_per_thread)
{ //each thread will enter here and will find the location for its own block,
    //will then stock it in the dedicated case for themselves in the entropy_location* entropy_loctaion_per_thread
    vec2 blk_location   = { 0 };
    uint8_t min_entropy = UINT8_MAX;
    uint8_t entropy_test;
    int idx;

    int block_size = blocks->block_side;
    int grid_size  = blocks->grid_side;

    int row   = blockIdx.y * blockDim.y + threadIdx.y; // just changed x and y here
    int col   = blockIdx.x * blockDim.x + threadIdx.x; // just changed x and y here
    int width = gridDim.x * blockDim.x;                // M = a* N

    //this gives the number of the case that thread starts to read from
    int th_idx_arr_location = (row * width + col) * block_size * block_size; //therefore it is not thread idx really
    int th_idx              = row * width + col;

    //Navigate through the block
    int keep_x;
    int keep_y;
    for (int block_x = 0; block_x < blocks->block_side; block_x++) {
        for (int block_y = 0; block_y < blocks->block_side; block_y++) {
            idx = th_idx_arr_location + block_x * block_size + block_y;
            dev_entropy_compute(blocks->states[idx], &entropy_test);
            if ((entropy_test < min_entropy) && (entropy_test > 1)) {
                min_entropy    = entropy_test;
                blk_location.x = block_x;
                blk_location.y = block_y;
                keep_x         = block_x;
                keep_y         = block_y;
            }
        }
    }

    int our_grid_Idx = blockIdx.x / grid_size;
    int our_grid_Idy = blockIdx.x % grid_size;

    entropy_loctaion_per_thread[th_idx].entropy            = min_entropy;
    entropy_loctaion_per_thread[th_idx].location_in_grid.x = our_grid_Idx;
    entropy_loctaion_per_thread[th_idx].location_in_grid.y = our_grid_Idy;
    entropy_loctaion_per_thread[th_idx].location_in_blk.x  = keep_x;
    entropy_loctaion_per_thread[th_idx].location_in_blk.y  = keep_y;

    //return new;
}

__global__ void
dev_blk_propagate(wfc_blocks_ptr blocks,
                  uint32_t gx, uint32_t gy,
                  uint64_t collapsed)
{
    /*if the thread is not owner the block with the minimum entropy it is not going to propagate*/
    int block_size   = blocks->block_side;
    int grid_size    = blocks->grid_side;
    int our_grid_Idx = blockIdx.x / grid_size;
    int our_grid_Idy = blockIdx.x % grid_size;
    if (our_grid_Idx != gx || our_grid_Idy != gy) {
        return; //do nothing and return
        //or should i synchronise the threads and they ar egoing to wiat for the executing thread?
    }
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;
    int width = gridDim.x * blockDim.x;

    //this gives the number of the case that thread starts to read from
    int th_idx_arr_location = (row * width + col) * block_size * block_size;
    int th_idx              = row * width + col;
    int idx;
    uint8_t bc;
    for (int i = 0; i < blocks->block_side; i++)
        for (int j = 0; j < blocks->block_side; j++) {
            idx = th_idx_arr_location + i * blocks->block_side + j;
            dev_bitfield_count(blocks->states[idx], &bc);

            if (bc != 1)
                blocks->states[idx] &= ~(collapsed);
        }
}

__global__ void //given grid indexes
dev_grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                         uint32_t x, uint32_t y, uint64_t collapsed)
{
    //each thread gets a block
    //for grid propopagate column it means that each thread will update the the its rows on that column
    //based on the given state
    //but not every grid is going to do this! only he grids that are on the same column
    //as the given grid is going to do this!!

    int block_size            = blocks->block_side;
    int grid_size             = blocks->grid_side;
    int our_grid_Idx          = blockIdx.x / grid_size;
    int our_grid_Idy          = blockIdx.x % grid_size;
    int row                   = blockIdx.y * blockDim.y + threadIdx.y;
    int col                   = blockIdx.x * blockDim.x + threadIdx.x;
    int width                 = gridDim.x * blockDim.x;
    int th_idx_arr_location   = (row * width + col) * block_size * block_size;
    int th_idx                = row * width + col;
    int grd_size_mod          = th_idx % grid_size; //=> will determine in which block we are
    int needed_grid_index     = gx * grid_size + gy;
    int needed_grid_index_mod = needed_grid_index % grid_size;
    if (grd_size_mod != needed_grid_index_mod) { //if not on the same column don't do somethin
        return;
    }

    if (th_idx == needed_grid_index) { //we have already propagated on this column in block
        return;
    }
    int idx;
    uint8_t bc;

    for (int i = 0; i < blocks->block_side; i++) {
        //for (int gxx = 0; gxx < blocks->grid_side; gxx++) { //dont traverse the grid side because each thread has a block, it does it naturally
        idx = th_idx_arr_location + i * block_size + y;
        dev_bitfield_count(blocks->states[idx], &bc);
        if (bc != 1)
            blocks->states[idx] &= ~(collapsed);
    }
}

__global__ void
dev_grd_propagate_row(wfc_blocks_ptr blocks,
                      uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                      uint64_t collapsed)
{
    int block_size            = blocks->block_side;
    int grid_size             = blocks->grid_side;
    int our_grid_Idx          = blockIdx.x / grid_size; //rox location in the grid
    int our_grid_Idy          = blockIdx.x % grid_size; //col location in the grid
    int row                   = blockIdx.y * blockDim.y + threadIdx.y;
    int col                   = blockIdx.x * blockDim.x + threadIdx.x;
    int width                 = gridDim.x * blockDim.x;
    int th_idx_arr_location   = (row * width + col) * block_size * block_size;
    int th_idx                = row * width + col;
    int grd_size_mod          = th_idx % grid_size; //=> will determine in which block we are
    int needed_grid_index     = gx * grid_size + gy;
    int needed_grid_index_mod = needed_grid_index % grid_size;
    int idx;
    uint8_t bc;

    //dont do if on a different row

    //dont do if the gridcudamemcpy is the asked grid
    if (th_idx == needed_grid_index) { //we have already propagated on this column in block
        return;
    }

    //do only if on the same row
    if (our_grid_Idx == gx) {
        for (int j = 0; j < blocks->block_side; j++) {
            idx = th_idx_arr_location + x * block_size + j;
            dev_bitfield_count(blocks->states[idx], &bc);
            if (bc != 1)
                blocks->states[idx] &= ~(collapsed);
        }
    }
    // return 0;
}

__device__ void //given grid indexes
dev_dev_grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                             uint32_t x, uint32_t y, uint64_t collapsed, int *pending, int *next)
{
    //each thread gets a block
    //for grid propopagate column it means that each thread will update the the its rows on that column
    //based on the given state
    //but not every grid is going to do this! only he grids that are on the same column
    //as the given grid is going to do this!!

    int block_size            = blocks->block_side;
    int grid_size             = blocks->grid_side;
    int our_grid_Idx          = blockIdx.x / grid_size;
    int our_grid_Idy          = blockIdx.x % grid_size;
    int row                   = blockIdx.y * blockDim.y + threadIdx.y;
    int col                   = blockIdx.x * blockDim.x + threadIdx.x;
    int width                 = gridDim.x * blockDim.x;
    int th_idx_arr_location   = (row * width + col) * block_size * block_size;
    int th_idx                = row * width + col;
    int grd_size_mod          = th_idx % grid_size; //=> will determine in which block we are
    int needed_grid_index     = gx * grid_size + gy;
    int needed_grid_index_mod = needed_grid_index % grid_size;
    if (grd_size_mod != needed_grid_index_mod) { //if not on the same column don't do somethin
        return;
    }

    if (th_idx == needed_grid_index) { //we have already propagated on this column in block
        return;
    }
    int idx;
    uint8_t bc;

    for (int i = 0; i < blocks->block_side; i++) {
        //for (int gxx = 0; gxx < blocks->grid_side; gxx++) { //dont traverse the grid side because each thread has a block, it does it naturally
        idx = th_idx_arr_location + i * block_size + y;
        dev_bitfield_count(blocks->states[idx], &bc);
        if (bc != 1) {
            blocks->states[idx] &= ~(collapsed);
            if (blocks->states[idx] == 1) {
                pending[(*next)] = idx;
                //(*next)++;
                atomicAdd(next, 1);
            }
        }
    }
}

__device__ void
dev_dev_blk_propagate(wfc_blocks_ptr blocks,
                      uint32_t gx, uint32_t gy,
                      uint64_t collapsed, int *pending, int *next)
{
    /*if the thread is not owner the block with the minimum entropy it is not going to propagate*/
    int block_size   = blocks->block_side;
    int grid_size    = blocks->grid_side;
    int our_grid_Idx = blockIdx.x / grid_size;
    int our_grid_Idy = blockIdx.x % grid_size;
    if (our_grid_Idx != gx || our_grid_Idy != gy) {
        return; //do nothing and return
        //or should i synchronise the threads and they ar egoing to wiat for the executing thread?
    }
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;
    int width = gridDim.x * blockDim.x;

    //this gives the number of the case that thread starts to read from
    int th_idx_arr_location = (row * width + col) * block_size * block_size;
    int th_idx              = row * width + col;
    int idx;
    uint8_t bc;
    for (int i = 0; i < blocks->block_side; i++)
        for (int j = 0; j < blocks->block_side; j++) {
            idx = th_idx_arr_location + i * blocks->block_side + j;
            dev_bitfield_count(blocks->states[idx], &bc);

            if (bc != 1) {
                blocks->states[idx] &= ~(collapsed);
                if (blocks->states[idx] == 1) {
                    pending[(*next)] = idx;
                    //(*next)++;
                    atomicAdd(next, 1);
                }
            }
        }
}

__device__ void
dev_set_mask(uint64_t grid_size, uint64_t block_size, masks *my_mask,
             uint64_t col_index, uint64_t row_index, uint64_t block_index,
             uint64_t collapsed, uint64_t *conflict_loc, uint64_t *th_loc, uint64_t* place, uint64_t* pass_collapsed)
{
    int our_grid_Idx        = blockIdx.x / grid_size; //rox location in the grid
    int our_grid_Idy        = blockIdx.x % grid_size; //col location in the grid
    int row                 = blockIdx.y * blockDim.y + threadIdx.y;
    int col                 = blockIdx.x * blockDim.x + threadIdx.x;
    int width               = gridDim.x * blockDim.x;
    int th_idx_arr_location = (row * width + col) * block_size * block_size;
    int th_idx              = row * width + col;
    int col_index_in_grid   = our_grid_Idy * grid_size + col_index % block_size;
    int row_index_in_grid   = our_grid_Idx * grid_size + row_index % block_size;

    *pass_collapsed = collapsed; 
    //if (th_idx != block_index) {
    //    return;
    //}
    //if (col_index != col_index_in_grid) {
    //    return;
    //}
    //if (row_index != row_index_in_grid) {
    //    return;
    //}

    uint64_t val_col   = my_mask->column_masks[col_index];
    uint64_t val_row   = my_mask->row_masks[row_index];
    uint64_t val_block = my_mask->block_masks[block_index];
    uint64_t ret_col   = val_col & ~(collapsed);
    uint64_t ret_row   = val_row & ~(collapsed);
    uint64_t ret_block = val_block & ~(collapsed);

    if (val_col != UINT64_MAX) //since in the beginning they are set to the maximum
        if (val_col == ret_col) {
            //printf("this value %lu has been set before on column idx: %lu\n", bit_to_val(collapsed), col_index);
            //printf("trying safe exit and pass to new seed if not in the load\n");
            *conflict_loc        = (uint64_t)col_index;
            *th_loc              = (uint64_t)th_idx_arr_location ;//th_idx;
            *place = 0; 
            (my_mask->safe_exit) = 1;
            return;
        }
    if (val_row != UINT64_MAX) //
        if (val_row == ret_row) {
            //printf("this value %lu has been set before on row idx: %lu\n", bit_to_val(collapsed), row_index);
            //printf("trying safe exit and pass to new seed if not in the load\n");
            *conflict_loc        = (uint64_t)row_index;
            *th_loc              = (uint64_t) th_idx;
            *place = 1; 
            (my_mask->safe_exit) = 1;
            return;
        }

    if (val_block != UINT64_MAX) //
        if (val_block == ret_block) {
            //printf("this value %lu has been set before on block idx: %lu\n", bit_to_val(collapsed), block_index);
            //printf("trying safe exit and pass to new seed if not in the load\n");
            *conflict_loc        = (uint64_t)block_index;
            *th_loc              = (uint64_t)th_idx;
            *place = 2; 
            (my_mask->safe_exit) = 1;
            return;
        }
    my_mask->column_masks[col_index]  = ret_col;
    my_mask->row_masks[row_index]     = ret_row;
    my_mask->block_masks[block_index] = ret_block;
}
__device__ void
dev_dev_grd_propagate_row(wfc_blocks_ptr blocks,
                          uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                          uint64_t collapsed, int *pending, int *next)
{
    int block_size            = blocks->block_side;
    int grid_size             = blocks->grid_side;
    int our_grid_Idx          = blockIdx.x / grid_size; //rox location in the grid
    int our_grid_Idy          = blockIdx.x % grid_size; //col location in the grid
    int row                   = blockIdx.y * blockDim.y + threadIdx.y;
    int col                   = blockIdx.x * blockDim.x + threadIdx.x;
    int width                 = gridDim.x * blockDim.x;
    int th_idx_arr_location   = (row * width + col) * block_size * block_size;
    int th_idx                = row * width + col;
    int grd_size_mod          = th_idx % grid_size; //=> will determine in which block we are
    int needed_grid_index     = gx * grid_size + gy;
    int needed_grid_index_mod = needed_grid_index % grid_size;
    int idx;
    uint8_t bc;

    //dont do if on a different row

    //dont do if the gridcudamemcpy is the asked grid
    if (th_idx == needed_grid_index) { //we have already propagated on this column in block
        return;
    }
    //do only if on the same row
    if (our_grid_Idx == gx) {
        for (int j = 0; j < blocks->block_side; j++) {
            idx = th_idx_arr_location + x * block_size + j;
            dev_bitfield_count(blocks->states[idx], &bc);
            if (bc != 1) {
                blocks->states[idx] &= ~(collapsed);
                if (blocks->states[idx] == 1) {
                    pending[(*next)] = idx;
                    //(*next)++;
                    atomicAdd(next, 1);
                }
            }
        }
    }
    // return 0;
}

__global__ void
global_propagate_all(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                     uint32_t x, uint32_t y, uint64_t collapsed, masks *my_masks, int *pending,
                     int *return_val, int *next, 
                     uint64_t* conflict_loc, uint64_t* th_loc, uint64_t* place, uint64_t* pass_collapsed )
{
    // size_t size = 2 * (blocks->grid_side * blocks->block_side - 1); // Row and Col
    // size += blocks->block_side * blocks->block_side - 1;            // Block

    //uint64_t *pending = (uint64_t*) calloc(size, sizeof(uint64_t));
    //int next          = 0;
    uint32_t col_idx;
    uint32_t row_idx;
    uint32_t block_idx;

    // BLOCK
    dev_dev_blk_propagate(blocks, gx, gy, collapsed, pending, next);

    //printf("After block propogate\n");
    //grd_print(NULL, blocks);

    //print_masks(my_masks, blocks->block_side, blocks->grid_side);

    // Column
    dev_dev_grd_propagate_column(blocks, gx, gy, x, y, collapsed, pending, next);

    // Row
    dev_dev_grd_propagate_row(blocks, gx, gy, x, y, collapsed, pending, next);

    // Chain reaction
    int gxx, gyy, xx, yy, new_idx;

    while (*next >= 0) {
        if (*next == 0 && pending[0] != -1) {
            new_idx = pending[0]; // Dequeue the next state

            yy  = new_idx % (blocks->block_side * blocks->block_side);
            xx  = new_idx / (blocks->block_side * blocks->block_side);
            gyy = xx % (blocks->grid_side);
            gxx = xx / (blocks->grid_side);

            int jsp  = gxx * (blocks->grid_side) + gyy;
            int jsp2 = jsp * (blocks->block_side * blocks->block_side);
            int jsp3 = new_idx - jsp2;

            yy = jsp3 % (blocks->grid_side);
            xx = jsp3 / (blocks->grid_side);

            col_idx   = gyy * blocks->block_side + yy;
            row_idx   = gxx * blocks->block_side + xx;
            block_idx = gxx * blocks->grid_side + gyy;
if(blockIdx.x == 0 && blockIdx.y == 0)
            dev_set_mask(blocks->grid_side, blocks->block_side, my_masks, col_idx, row_idx, block_idx, blocks->states[new_idx], conflict_loc, th_loc, place, pass_collapsed);
            __syncthreads();

            if (my_masks->safe_exit == 1) {
                //printf("conflict while collapsing, safe exiting\n");
                *return_val = 1;
            }

            // Instead of making a recursive call, add the new states to the queue
            dev_dev_blk_propagate(blocks, gxx, gyy, blocks->states[new_idx], pending, next);
            dev_dev_grd_propagate_column(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, next);
            dev_dev_grd_propagate_row(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, next);
        }

        new_idx = pending[(--(*next))]; // Dequeue the next state

        yy  = new_idx % (blocks->block_side * blocks->block_side);
        xx  = new_idx / (blocks->block_side * blocks->block_side);
        gyy = xx % (blocks->grid_side);
        gxx = xx / (blocks->grid_side);

        int jsp  = gxx * (blocks->grid_side) + gyy;
        int jsp2 = jsp * (blocks->block_side * blocks->block_side);
        int jsp3 = new_idx - jsp2;

        yy = jsp3 % (blocks->grid_side);
        xx = jsp3 / (blocks->grid_side);

        col_idx   = gyy * blocks->block_side + yy;
        row_idx   = gxx * blocks->block_side + xx;
        block_idx = gxx * blocks->grid_side + gyy;
if(blockIdx.x == 0 && blockIdx.y == 0)

        dev_set_mask(blocks->grid_side, blocks->block_side, my_masks, col_idx, row_idx, block_idx, blocks->states[new_idx], conflict_loc, th_loc, place, pass_collapsed);
        __syncthreads();

        if (my_masks->safe_exit == 1) {
            //printf("conflict while collapsing, safe exiting\n");
            *return_val = 1;
        }

        // Instead of making a recursive call, add the new states to the queue
        dev_dev_blk_propagate(blocks, gxx, gyy, blocks->states[new_idx], pending, next);
        dev_dev_grd_propagate_column(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, next);
        dev_dev_grd_propagate_row(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, next);
    }
    //}

    *return_val = 0;
}
