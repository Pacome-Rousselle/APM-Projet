//#define _GNU_SOURCE

#include "wfc.h"
#include "bitfield.h"
// #include "utils.h"
#include "md5.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <strings.h>

#define restrict  

#include <math.h> //for pow
#include <omp.h>


entropy_location
blk_min_entropy_omp(const wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    vec2 blk_location   = { 0 };
    uint8_t min_entropy = UINT8_MAX;
    uint8_t entropy_test;
    int idx;
    //Navigate through the block
    for (int block_x = 0; block_x < blocks->block_side; block_x++)
        for (int block_y = 0; block_y < blocks->block_side; block_y++) {
            idx          = get_thread_glob_idx(blocks, gx, gy, block_x, block_y);
            entropy_test = entropy_compute(blocks->states[idx]);
            //if(entropy_test < min_entropy)
            if ((entropy_test < min_entropy) && (entropy_test > 1)) {
                min_entropy    = entropy_test;
                blk_location.x = block_x;
                blk_location.y = block_y;
            }
        }
    entropy_location new_loc;
    new_loc.entropy            = min_entropy;
    new_loc.location_in_blk    = blk_location;
    new_loc.location_in_grid.x = gx;
    new_loc.location_in_grid.y = gy;

    return new_loc;
}

void
blk_propagate_omp(wfc_blocks_ptr blocks,
              uint32_t gx, uint32_t gy,
              uint64_t collapsed, uint64_t *pending, int *next, masks* my_masks)
{
    int idx;
    uint64_t bc;
    #pragma omp parallel num_threads(12)
    {
        //if (omp_get_thread_num() == 0)
printf("omp num threads %d \n", omp_get_num_threads()); 
        
        for (int i = 0; i < blocks->block_side; i++)
        for (int j = 0; j < blocks->block_side; j++) {
            idx = get_thread_glob_idx(blocks, gx, gy, i, j);

            if (bitfield_count(blocks->states[idx]) != 1) {
                blocks->states[idx] &= ~(collapsed);
                if (entropy_compute(blocks->states[idx]) == 1) {
                    pending[*next] = idx;
                    (*next)++;
                }
            }
        }
    }
}

// Traverse the row to propagate,
//aka ridding every other cases of the collapsed state
void
grd_propagate_row_omp(wfc_blocks_ptr blocks,
                  uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                  uint64_t collapsed, uint64_t *pending, int *next, masks* my_mask)
{
    //propopgate only on the column
    //stay in the same row
    //change the columns
    int idx;
    uint64_t bc;
    
    for (int j = 0; j < blocks->block_side; j++)
        for (int gyy = 0; gyy < blocks->grid_side; gyy++) {
            //if( gyy != gy){
            idx = get_thread_glob_idx(blocks, gx, gyy, x, j);
            if (bitfield_count(blocks->states[idx]) != 1) {
                blocks->states[idx] &= ~(collapsed);
                if (entropy_compute(blocks->states[idx]) == 1) {
                    pending[*next] = idx;
                    (*next)++;
                }
            }
        }
    // return 0;
}

// Traverse the column to propagate, aka ridding every other cases of the collapsed state
// gy and y (grid and blocks row) are fixed
void
grd_propagate_column_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                     uint32_t x, uint32_t y, uint64_t collapsed, uint64_t *pending, int *next, masks* my_mask)
{
    //stay in the same column
    //change the rows
    int idx;

    for (int i = 0; i < blocks->block_side; i++)
        for (int gxx = 0; gxx < blocks->grid_side; gxx++) {
            idx = get_thread_glob_idx(blocks, gxx, gy, i, y);
            if (bitfield_count(blocks->states[idx]) != 1) {
                blocks->states[idx] &= ~(collapsed);
                if (entropy_compute(blocks->states[idx]) == 1) {
                    pending[*next] = idx;
                    (*next)++;
                }
            }
        }
}

void
/*sets the masks and checks if the value has been used before */
set_mask_omp(masks *my_mask, uint64_t col_index, uint64_t row_index, uint64_t block_index, uint64_t collapsed)
{   //s_f = success failure value
    // if it is 0 it is success if it is 1 it is failure

    uint64_t val_col   = my_mask->column_masks[col_index];
    uint64_t val_row   = my_mask->row_masks[row_index];
    uint64_t val_block = my_mask->block_masks[block_index];
    uint64_t ret_col   = val_col & ~(collapsed);
    uint64_t ret_row   = val_row & ~(collapsed);
    uint64_t ret_block = val_block & ~(collapsed);

    if (val_col != UINT64_MAX) //since in the beginning they are set to the maximum
        if (val_col == ret_col) {
            printf("this value %lu has been set before on column idx: %lu\n", bit_to_val(collapsed), col_index);
            printf("trying safe exit and pass to new seed if not in the load\n");
            (my_mask->safe_exit) = 1;
            return;
        }
    if (val_row != UINT64_MAX) //
        if (val_row == ret_row) {
            printf("this value %lu has been set before on row idx: %lu\n", bit_to_val(collapsed), row_index);
            printf("trying safe exit and pass to new seed if not in the load\n");

            (my_mask->safe_exit) = 1;
            return;
        }

    if (val_block != UINT64_MAX) //
        if (val_block == ret_block) {
            printf("this value %lu has been set before on block idx: %lu\n", bit_to_val(collapsed), block_index);
            printf("trying safe exit and pass to new seed if not in the load\n");

            (my_mask->safe_exit) = 1;
            return;
        }
    my_mask->column_masks[col_index] = ret_col;
    my_mask->row_masks[row_index]    = ret_row;
    my_mask->block_masks[block_index]  = ret_block;
}
bool
propagate_all_omp(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
              uint32_t x, uint32_t y, uint64_t collapsed, masks* my_masks)
{
    size_t size = 2 * (blocks->grid_side * blocks->block_side - 1); // Row and Col
    size += blocks->block_side * blocks->block_side - 1;            // Block

    uint64_t *pending = (uint64_t*) calloc(size, sizeof(uint64_t));
    int next          = 0;
    uint32_t col_idx  ;
    uint32_t row_idx  ;
    uint32_t block_idx;

    // BLOCK
    blk_propagate_omp(blocks, gx, gy, collapsed, pending, &next, my_masks);

   // printf("After block propogate\n");
   // grd_print(NULL, blocks);

    //print_masks(my_masks, blocks->block_side, blocks->grid_side); 

   // if (grd_check_error_in_blk(blocks, gx, gy, x, y) == false) {
   //     //printf("encountered another state same value in same block\n");
   //     return false;
   // };

    // Column
    grd_propagate_column_omp(blocks, gx, gy, x, y, collapsed, pending, &next, my_masks);

    //printf("After column\n");
    //grd_print(NULL, blocks);

    //if (grd_check_error_in_column(blocks, gy, y) == false) {
    //    //printf("encountered another state same value in same column\n");
    //    return false;
    //};

    // Row
    grd_propagate_row_omp(blocks, gx, gy, x, y, collapsed, pending, &next,  my_masks);

    //printf("After row\n");
    //grd_print(NULL, blocks);

    //if (grd_check_error_in_row(blocks, gx, x) == false) {
    //    //printf("encountered another state same value in same row\n");
    //    return false;
    //};

    // Chain reaction
    int gxx, gyy, xx, yy, new_idx;

    for (int i = 0; i < next; i++) {
        new_idx = pending[i];
        //printf("In pending %d\n",new_idx);
        yy = new_idx % (blocks->block_side * blocks->block_side);
        xx = new_idx / (blocks->block_side * blocks->block_side);
        //printf("%d/%d = %d\n",new_idx,blocks->block_side*blocks->block_side, xx);
        gyy = xx % (blocks->grid_side);
        //printf("%d modulo %d = %d\n",xx,blocks->grid_side, gyy);
        gxx = xx / (blocks->grid_side);

        int jsp = gxx * (blocks->grid_side) + gyy;

        int jsp2 = jsp * (blocks->block_side * blocks->block_side);

        int jsp3 = new_idx - jsp2;

        yy = jsp3 % (blocks->grid_side);
        xx = jsp3 / (blocks->grid_side);

        col_idx   = gyy * blocks->block_side + yy; 
        row_idx   = gxx * blocks->block_side + xx; 
        block_idx = gxx * blocks->grid_side + gyy; 
        //printf("before setting masks in pending\n"); 
        //print_masks(my_masks, blocks->block_side, blocks->grid_side); 
        set_mask_omp(my_masks, col_idx, row_idx, block_idx, blocks->states[new_idx]); 
        if(my_masks->safe_exit == 1)
        {
            printf("while in pending states saw a conflict\n"); 
            return false; 
        }
        //printf("%d %d %d %d\n", gxx, gyy, xx, yy);
        //printf("%d*%d+%d = %d*%d+%d = %d\n",gxx,blocks->grid_side,gyy,xx,blocks->block_side*blocks->block_side,yy,new_idx);
        propagate_all_omp(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], my_masks);
    }

    return true;
}



/*bool propagate_all(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                   uint32_t x, uint32_t y, uint64_t collapsed, masks* my_masks)
{
    size_t size = 2 * (blocks->grid_side * blocks->block_side - 1); // Row and Col
    size += blocks->block_side * blocks->block_side - 1;            // Block

    uint64_t *pending = (uint64_t*) calloc(size, sizeof(uint64_t));
    int next          = 0;
    uint32_t col_idx  ;
    uint32_t row_idx  ;
    uint32_t block_idx;

    // BLOCK
    blk_propagate(blocks, gx, gy, collapsed, pending, &next, my_masks);

    //printf("After block propogate\n");
    //grd_print(NULL, blocks);

    //print_masks(my_masks, blocks->block_side, blocks->grid_side); 

    // Column
    grd_propagate_column(blocks, gx, gy, x, y, collapsed, pending, &next, my_masks);

    // Row
    grd_propagate_row(blocks, gx, gy, x, y, collapsed, pending, &next,  my_masks);

    // Chain reaction
    int gxx, gyy, xx, yy, new_idx;

    while (next > 0) {
        new_idx = pending[--next];  // Dequeue the next state

        yy = new_idx % (blocks->block_side * blocks->block_side);
        xx = new_idx / (blocks->block_side * blocks->block_side);
        gyy = xx % (blocks->grid_side);
        gxx = xx / (blocks->grid_side);

        int jsp = gxx * (blocks->grid_side) + gyy;
        int jsp2 = jsp * (blocks->block_side * blocks->block_side);
        int jsp3 = new_idx - jsp2;

        yy = jsp3 % (blocks->grid_side);
        xx = jsp3 / (blocks->grid_side);

        col_idx   = gyy * blocks->block_side + yy; 
        row_idx   = gxx * blocks->block_side + xx; 
        block_idx = gxx * blocks->grid_side + gyy; 
        
        set_mask(my_masks, col_idx, row_idx, block_idx, blocks->states[new_idx]); 
        if(my_masks->safe_exit == 1)
        {
            printf("conflict while collapsing, safe exiting\n"); 
            return false; 
        }

        // Instead of making a recursive call, add the new states to the queue
        blk_propagate(blocks, gxx, gyy, blocks->states[new_idx], pending, &next, my_masks);
        grd_propagate_column(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, &next, my_masks);
        grd_propagate_row(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, &next, my_masks);
    }

    return true;
}*/

/*bool propagate_all(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                   uint32_t x, uint32_t y, uint64_t collapsed, masks* my_masks)
{
    size_t size = 2 * (blocks->grid_side * blocks->block_side - 1); // Row and Col
    size += blocks->block_side * blocks->block_side - 1;            // Block

    uint64_t *pending = (uint64_t*) calloc(size, sizeof(uint64_t));
    int next          = 0;
    uint32_t col_idx  ;
    uint32_t row_idx  ;
    uint32_t block_idx;

    // BLOCK
    blk_propagate(blocks, gx, gy, collapsed, pending, &next, my_masks);

    printf("After block propogate\n");
    grd_print(NULL, blocks);

    print_masks(my_masks, blocks->block_side, blocks->grid_side); 

    // Column
    grd_propagate_column(blocks, gx, gy, x, y, collapsed, pending, &next, my_masks);

    // Row
    grd_propagate_row(blocks, gx, gy, x, y, collapsed, pending, &next,  my_masks);

    // Chain reaction
    int gxx, gyy, xx, yy, new_idx;

    for (int i = next-1; i >= 0; i--) {
        new_idx = pending[i];

        yy = new_idx % (blocks->block_side * blocks->block_side);
        xx = new_idx / (blocks->block_side * blocks->block_side);
        gyy = xx % (blocks->grid_side);
        gxx = xx / (blocks->grid_side);

        int jsp = gxx * (blocks->grid_side) + gyy;
        int jsp2 = jsp * (blocks->block_side * blocks->block_side);
        int jsp3 = new_idx - jsp2;

        yy = jsp3 % (blocks->grid_side);
        xx = jsp3 / (blocks->grid_side);

        col_idx   = gyy * blocks->block_side + yy; 
        row_idx   = gxx * blocks->block_side + xx; 
        block_idx = gxx * blocks->grid_side + gyy; 
        
        set_mask(my_masks, col_idx, row_idx, block_idx, blocks->states[new_idx]); 
        if(my_masks->safe_exit == 1)
        {
            printf("conflict while collapsing, safe exiting\n"); 
            return false; 
        }

        // Instead of making a recursive call, add the new states to the queue
        blk_propagate(blocks, gxx, gyy, blocks->states[new_idx], pending, &next, my_masks);
        grd_propagate_column(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, &next, my_masks);
        grd_propagate_row(blocks, gxx, gyy, xx, yy, blocks->states[new_idx], pending, &next, my_masks);
    }

    return true;
}*/


// When propagating, check if a state gets only 1 state left to propagate it further
// Traverse the block to propagate, aka ridding every other cases of the collapsed state
// Make a loop to traverse all the grids, block by block
/*void
blk_propagate(wfc_blocks_ptr blocks,
              uint32_t gx, uint32_t gy,
              uint64_t collapsed)
{
    int idx;
    uint64_t bc;
    for (int i = 0; i < blocks->block_side; i++)
        for (int j = 0; j < blocks->block_side; j++) {
            idx = get_thread_glob_idx(blocks, gx, gy, i, j);

            // Bit wise AND (&=) with inverse of collapsed (~)
            //(all 1s except the state at 0 we want to collapse)

            bc = bitfield_count(blocks->states[idx]);
            //printf("bc = %lu \n", bc);
            if (bc != 1) //if this is not present it sets the block to zero but if in a for casse les perfs;
                         //what to do?
                blocks->states[idx] &= ~(collapsed);
        }
}

// Traverse the row to propagate,
//aka ridding every other cases of the collapsed state
void
grd_propagate_row(wfc_blocks_ptr blocks,
                  uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                  uint64_t collapsed)
{
    //propopgate only on the column
    //stay in the same row
    //change the columns
    int idx;
    uint64_t bc;
    for (int j = 0; j < blocks->block_side; j++)
        for (int gyy = 0; gyy < blocks->grid_side; gyy++) {
            //if( gyy != gy){
            idx = get_thread_glob_idx(blocks, gx, gyy, x, j);

            // Bit wise AND (&=) with inverse of collapsed (~)
            //(all 1s except the state at 0 we want to collapse)

            bc = bitfield_count(blocks->states[idx]);
            //printf("bc = %lu \n", bc);
            if (bc != 1) //if this is not present it sets the block to zero but if in a for casse les perfs;
                         //what to do?
                blocks->states[idx] &= ~(collapsed);
            //}
        }
    // return 0;
}

// Traverse the column to propagate, aka ridding every other cases of the collapsed state
void
grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                     uint32_t x, uint32_t y, uint64_t collapsed)
{
    //stay in the same column
    //change the rows
    int idx;
    uint8_t bc;

    for (int i = 0; i < blocks->block_side; i++)
        for (int gxx = 0; gxx < blocks->grid_side; gxx++) {
            idx = get_thread_glob_idx(blocks, gxx, gy, i, y);
            // Bit wise AND (&=) with inverse of collapsed (~)
            //(all 1s except the state at 0 we want to collapse)
            bc = bitfield_count(blocks->states[idx]);
            //to remove once we resolve the thing with minimum entropy
            if (bc != 1) //if this is not present it sets the block to zero but if in a for casse les perfs;
                         //what to do?
                blocks->states[idx] &= ~(collapsed);
        }
    //return 0;
}*/

// Printing functions
//void blk_print(FILE *const, const wfc_blocks_ptr block, uint32_t gx, uint32_t gy)
//{}

/*void
printBinary2(uint64_t number)
{
    // Determine the number of bits in u64_t
    int numBits = sizeof(uint64_t) + 1;

    // Loop through each bit in the number, starting from the most significant bit
    for (int i = numBits - 1; i >= 0; i--) {
        // Use a bitwise AND operation to check the value of the current bit
        if ((number & (1ULL << i)) != 0) {
            printf("1");
        } else {
            printf("0");
        }
    }
    // printf("\n");
}*/
