#define _GNU_SOURCE

#include "wfc.h"
#include "bitfield.h"
#include "md5.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <strings.h>

#include <math.h> //for pow

uint64_t
entropy_collapse_state(uint64_t state,
                       uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                       uint64_t seed,
                       uint64_t iteration)
{
    uint8_t digest[16]     = { 0 };
    uint64_t random_number = 0;
    struct {
        uint32_t gx, gy, x, y;
        uint64_t seed, iteration;
    } random_state = {
        .gx        = gx,
        .gy        = gy,
        .x         = x,
        .y         = y,
        .seed      = seed,
        .iteration = iteration,
    };
    // Digest is now randomly filled
    md5((uint8_t *)&random_state, sizeof(random_state), digest);
    // Choose a random bit to set state to
    //random_number = bitfield_count(state)%digest[0];
    random_number = digest[0] % bitfield_count(state);

    state = bitfield_only_nth_set(state, random_number);
    return state;
}

uint8_t
entropy_compute(uint64_t state)
{
    return bitfield_count(state);
}

void
wfc_clone_into(wfc_blocks_ptr *const restrict ret_ptr, uint64_t seed, const wfc_blocks_ptr blocks)
{
    const uint64_t grid_size  = blocks->grid_side;
    const uint64_t block_size = blocks->block_side;
    wfc_blocks_ptr ret        = *ret_ptr;

    const uint64_t size = (wfc_control_states_count(grid_size, block_size) * sizeof(uint64_t)) +
                          (grid_size * grid_size * block_size * block_size * sizeof(uint64_t)) +
                          sizeof(wfc_blocks);

    if (NULL == ret) {
        if (NULL == (ret = malloc(size))) {
            fprintf(stderr, "failed to clone blocks structure\n");
            exit(EXIT_FAILURE);
        }
    } else if (grid_size != ret->grid_side || block_size != ret->block_side) {
        fprintf(stderr, "size mismatch!\n");
        exit(EXIT_FAILURE);
    }

    memcpy(ret, blocks, size);
    // ret->states[0] = seed;

    uint64_t val = convertToBits(seed);
    ret->states[0] = val;

    *ret_ptr       = ret;
}

entropy_location
blk_min_entropy(const wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
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
    entropy_location new;
    new.entropy            = min_entropy;
    new.location_in_blk    = blk_location;
    new.location_in_grid.x = gx;
    new.location_in_grid.y = gy;

    return new;
}

static inline uint64_t
blk_filter_mask_for_column(wfc_blocks_ptr blocks,
                           uint32_t gy, uint32_t y,
                           uint64_t collapsed)
{
    return 0;
}

static inline uint64_t
blk_filter_mask_for_row(wfc_blocks_ptr blocks,
                        uint32_t gx, uint32_t x,
                        uint64_t collapsed)
{
    return 0;
}

static inline uint64_t
blk_filter_mask_for_block(wfc_blocks_ptr blocks,
                          uint32_t gy, uint32_t gx,
                          uint64_t collapsed)
{
    return 0;
}

bool
grd_check_error_in_row(wfc_blocks_ptr blocks, uint32_t gx, uint32_t x)
{
    printf("in row error checking \n"); 
    int idx;
    int idx_other;
    uint64_t cell_state;
    uint8_t cell_entropy;

    uint8_t other_cell_entropy;
    uint64_t other_cell_state;

    for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
        for (uint32_t y = 0; y < blocks->block_side; y++) {
            idx          = get_thread_glob_idx(blocks, gx, gy, x, y);
            cell_state   = blocks->states[idx];
            cell_entropy = entropy_compute(cell_state);

            if (cell_entropy == 1) {

                for (uint32_t gyy = 0; gyy < blocks->grid_side; gyy++) {
                    for (uint32_t yy = 0; yy < blocks->block_side; yy++) {
                        idx_other = get_thread_glob_idx(blocks, gx, gyy, x, yy);
                        if (idx != idx_other) {
                            other_cell_entropy = entropy_compute(blocks->states[idx_other]);
                            if (other_cell_entropy == 1) {
                                other_cell_state = blocks->states[idx_other];
                                if (other_cell_state == cell_state) {
                                    return false;
                                }
                            }
                        }
                    }
                }

            }
        }
    }
    // No errors found
    return true;
}

bool
grd_check_error_in_column(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y)
{
    printf("in col error checking \n"); 
    int idx;
    int idx_other;
    uint64_t cell_state;
    uint8_t cell_entropy;

    uint8_t other_cell_entropy;
    uint64_t other_cell_state;

    for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
        for (uint32_t x = 0; x < blocks->block_side; x++) {
            idx          = get_thread_glob_idx(blocks, gx, gy, x, y);
            cell_state   = blocks->states[idx];
            cell_entropy = entropy_compute(cell_state);

            if (cell_entropy == 1) {

                for (uint32_t gxx = 0; gxx < blocks->grid_side; gxx++) {
                    for (uint32_t xx = 0; xx < blocks->block_side; xx++) {
                        idx_other = get_thread_glob_idx(blocks, gxx, gy, xx, y);
                        if (idx != idx_other) {
                            other_cell_entropy = entropy_compute(blocks->states[idx_other]);
                            if (other_cell_entropy == 1) {
                                other_cell_state = blocks->states[idx_other];
                                if (other_cell_state == cell_state) {
                                    return false;
                                }
                            }
                        }
                    }
                }

            }
        }
    }
    // No errors found
    return true;
}

bool
grd_check_error_in_blk(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    int idx;
    int idx_other;
    uint64_t cell_state;
    uint8_t cell_entropy;

    uint8_t other_cell_entropy;
    uint64_t other_cell_state;

    for (int i = 0; i < blocks->block_side; i++)
        for (int j = 0; j < blocks->block_side; j++) {
            idx          = get_thread_glob_idx(blocks, gx, gy, x, y);
            cell_state   = blocks->states[idx];
            cell_entropy = entropy_compute(cell_state);

            if (cell_entropy == 1)
                for (int ii = 0; ii < blocks->block_side; ii++)
                    for (int jj = 0; jj < blocks->block_side; jj++) {
                        idx_other = get_thread_glob_idx(blocks, gx, gy, ii, jj);
                        other_cell_state = blocks->states[idx_other];
                        other_cell_entropy = entropy_compute(blocks->states[idx_other]);

                        if (idx != idx_other && other_cell_entropy == 1 && other_cell_state == cell_state)  
                            return false;
                    }   
        }
    return true;
}

// When propagating, check if a state gets only 1 state left to propagate it further
// Traverse the block to propagate, aka ridding every other cases of the collapsed state
// Make a loop to traverse all the grids, block by block
void
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
// gy and y (grid and blocks row) are fixed
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
}

void
all_propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, 
              uint32_t x, uint32_t y, uint64_t collapsed)
{
    blk_propagate(blocks,gx,gy,collapsed);
    grd_propagate_column(blocks,gx,gy,x,y,collapsed);
    grd_propagate_row(blocks,gx,gy,x,y,collapsed);
}

// Printing functions
//void blk_print(FILE *const, const wfc_blocks_ptr block, uint32_t gx, uint32_t gy)
//{}

void
printBinary2(uint64_t number)
{
    // Determine the number of bits in u64_t
    int numBits = sizeof(uint64_t) + 1;

    // Loop through each bit in the number, starting from the most significant bit
    if (entropy_compute(number) == 1)
    {
        int pow = 1;
        while (number != 1)
        {
            number/=2;
            pow++;
        }
        
        printf("    %d    ", pow);
        return;
    }
    
    for (int i = numBits - 1; i >= 0; i--) {
        // Use a bitwise AND operation to check the value of the current bit
        if ((number & (1ULL << i)) != 0) {
            printf("1");
        } else {
            printf("0");
        }
    }
    // printf("\n");
}

void
grd_print(FILE *const file, const wfc_blocks_ptr block)
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
                    //printf("%dgx:%ugy:%ux:%uy:%u",
                    //get_thread_glob_idx(block, ii, i, jj, j), ii, i, jj, j);

                    printBinary2(collapsed);
                    fprintf(fp, " |", collapsed);
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