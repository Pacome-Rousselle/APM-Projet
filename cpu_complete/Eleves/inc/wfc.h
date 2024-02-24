#pragma once

#include "types.h"

#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define forever for (;;)

/// Parses the arguments, prints the help message if needed and abort on error.
wfc_args wfc_parse_args(int argc, char **argv);

/// Get the next seed to try. If there are no more seeds to try, it will exit the process.
bool try_next_seed(seeds_list *restrict *const seeds, uint64_t *restrict return_seed);

/// Count the total number of seeds.
uint64_t count_seeds(const seeds_list *restrict const seeds);

/// Load the positions from a file. You must free the thing yourself. On error
/// kill the program.
//wfc_blocks_ptr wfc_load(uint64_t, const char *);
wfc_load_returns* 
wfc_load(uint64_t, const char *);



/// Clone the blocks structure. You must free the return yourself.
void wfc_clone_into(wfc_blocks_ptr *const restrict ret_ptr, uint64_t, const wfc_blocks_ptr);

void my_masks_clone(masks_ptr *const restrict my_masks_ret_ptr,const masks_ptr orig_masks , wfc_blocks_ptr init); 

/// Save the grid to a folder by creating a new file or overwrite it, on error kills the program.
void wfc_save_into(const wfc_blocks_ptr, const char data[], const char folder[]);

static inline uint64_t
wfc_control_states_count(uint64_t grid_size, uint64_t block_size)
{
    return 0;
}

static inline uint64_t *
grd_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    return 0;
}

static inline uint64_t *
blk_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
        // Calculate the index into the states array for the given coordinates
    uint64_t index = (gx * blocks->grid_side + gy) * (blocks->block_side * blocks->block_side)
                                                     + (x * blocks->block_side + y);
    
    
    
    //printf("index %d ", index); 
    // Return the address of the block at the calculated index

    return &(blocks->states[index]);
    //return 0;
}

// Printing functions
void blk_print(FILE *const, const wfc_blocks_ptr block, uint32_t gx, uint32_t gy);
void grd_print(FILE *const, const wfc_blocks_ptr block);

// Entropy functions
entropy_location blk_min_entropy(const wfc_blocks_ptr block, uint32_t gx, uint32_t gy);
uint8_t entropy_compute(uint64_t);
uint64_t entropy_collapse_state(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t);

// Propagation functions
void blk_propagate(wfc_blocks_ptr, uint32_t, uint32_t, uint64_t, uint64_t*, int*, masks* my_mask);
void grd_propagate_column(wfc_blocks_ptr, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t*, int*, masks* my_mask);
void grd_propagate_row(wfc_blocks_ptr, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t*, int*, masks* my_mask);
bool propagate_all(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t collapsed,
                     masks* my_mask);

// Check functions
bool grd_check_error_in_row(wfc_blocks_ptr blocks, uint32_t gx, uint32_t x);
bool grd_check_error_in_column(wfc_blocks_ptr blocks, uint32_t gy, uint32_t y);
bool grd_check_error_in_blk(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y);


// Solvers
bool solve_cpu(wfc_blocks_ptr,    uint64_t seed, masks*);
bool solve_openmp(wfc_blocks_ptr, uint64_t seed, masks*);
bool solve_target(wfc_blocks_ptr, uint64_t seed, masks*);
#if defined(WFC_CUDA)
bool solve_cuda(wfc_blocks_ptr,  uint64_t seed, masks*);
#endif

static const wfc_solver solvers[] = {
    { "cpu", solve_cpu },
    { "omp", solve_openmp },
    { "target", solve_target },
#if defined(WFC_CUDA)
    { "cuda", solve_cuda },
#endif
};

// Get a global thread index
static inline int get_thread_glob_idx(const wfc_blocks_ptr blocks, uint8_t grid_X, uint8_t grid_Y, uint8_t block_X, uint8_t block_Y)
{
    int block_id = grid_X * blocks->grid_side + grid_Y;
    int Nb_threads = blocks->block_side * blocks->block_side;
    int local_id = block_X * blocks->block_side + block_Y;
    return block_id*Nb_threads + local_id;
};

/*masks*/
void print_masks(masks* my_masks, uint64_t block_size, uint64_t grid_size);
void set_mask(masks* my_mask, uint64_t col_index,uint64_t row_index,  
                uint64_t block_index,uint64_t collapsed );  //if it can not set the flag it will return a boolean type int to recover
uint64_t convertToBits(uint64_t value); 
uint64_t bit_to_val(uint64_t value);