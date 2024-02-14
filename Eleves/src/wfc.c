#define _GNU_SOURCE

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
    random_number = bitfield_count(state)%digest[0];
    bitfield_set(state,random_number);
    return 0;
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
    ret->states[0] = seed;
    *ret_ptr       = ret;
}

entropy_location
blk_min_entropy(const wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    vec2 blk_location   = { 0 };
    vec2 grid_location   = { 0 };
    uint8_t min_entropy = UINT8_MAX;
    uint8_t entropy_test;
    int idx;
    //Navigate through the grid
    for (int grid_x = 0; grid_x < gx; grid_x++)
        for (int grid_y = 0; grid_y < gy; grid_y++)
        //Navigate through the block
            for (int block_x = 0; block_x < blocks->block_side; block_x++)
                for (int block_y = 0; block_y < blocks->block_side; block_y++)
                {
                    idx = get_thread_glob_idx(blocks, grid_x,grid_y,block_x,block_y);
                    entropy_test = entropy_compute(blocks->states[idx]);
                    if(entropy_test < min_entropy)
                    {
                        min_entropy = entropy_test;
                        blk_location.x = block_x;
                        blk_location.y = block_y;

                        grid_location.x = grid_x;
                        grid_location.y = grid_y;
                    }
                }
    entropy_location new;
    new.entropy = min_entropy;
    new.location_in_blk = blk_location;
    new.location_in_grid = grid_location;

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
grd_check_error_in_column(wfc_blocks_ptr blocks, uint32_t gx)
{
    return 0;
}

// Traverse the block to propagate, aka ridding every other cases of the collapsed state
void
blk_propagate(wfc_blocks_ptr blocks,
              uint32_t gx, uint32_t gy,
              uint64_t collapsed)
{
    for (int i = 0; i < gx; i++)
        for (int j = 0; j < gy; j++)
        {
            blocks->states[collapsed];
        }
        
    
    return 0;
}

// Traverse the row to propagate, aka ridding every other cases of the collapsed state
void
grd_propagate_row(wfc_blocks_ptr blocks,
                  uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                  uint64_t collapsed)
{
    return 0;
}

// Traverse the column to propagate, aka ridding every other cases of the collapsed state
void
grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                     uint32_t x, uint32_t y, uint64_t collapsed)
{
    return 0;
}

// Printing functions
void blk_print(FILE *const, const wfc_blocks_ptr block, uint32_t gx, uint32_t gy)
{}
void grd_print(FILE *const, const wfc_blocks_ptr block)
{}