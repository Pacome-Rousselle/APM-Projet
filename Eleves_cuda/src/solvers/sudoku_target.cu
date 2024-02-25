//#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

bool
solve_target(wfc_blocks_ptr blocks, uint64_t seed, masks* my_masks)
{
   //printf("inside solver cpu \n"); 
        //print_masks(my_masks, blocks->block_side, blocks->grid_side);
    //grd_print(NULL, blocks);
    // seed = blocks->states[0];

    uint64_t iteration  = 0;

    entropy_location loc, test;
    int changed = 0; 
    /*printing the current stuation in the beginning*/

    //printf("grid side: %u\n", blocks->grid_side); 
    //printf("block side: %u\n", blocks->block_side); 
    
uint32_t col_idx  ;
uint32_t row_idx  ;
uint32_t block_idx;
    forever {
        //printf("Itération %d\n",iteration);
        //while find_min_entropy in grid 
        loc.entropy = UINT8_MAX;
        for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++)
            {
                test = blk_min_entropy(blocks,gx,gy);
                if(test.entropy < loc.entropy)
                    loc = test;
            }
        if (loc.entropy == UINT8_MAX)
            break;        
        
        //printf("loc.entropy is %zu\n", loc.entropy); 
        if (loc.entropy == 1)
        {
        //printf("inside loc.entropy = 1\n"); 
            return false;
        }
        //printf("before collapsing\n"); 
        //grd_print(NULL, blocks); 
        //print_masks(my_masks, blocks->block_side, blocks->grid_side);

        // Collapse
        int idx = get_thread_glob_idx(blocks,
                                      loc.location_in_grid.x,
                                      loc.location_in_grid.y,
                                      loc.location_in_blk.x,
                                      loc.location_in_blk.y);
        //printf("idx: %d\n", idx); 
        blocks->states[idx] = entropy_collapse_state(blocks->states[idx],
                                                    loc.location_in_grid.x,
                                                    loc.location_in_grid.y,
                                                    loc.location_in_blk.x,
                                                    loc.location_in_blk.y,
                                                    seed,
                                                    iteration);
        col_idx   = loc.location_in_grid.y * blocks->block_side + loc.location_in_blk.y; 
        row_idx   = loc.location_in_grid.x * blocks->block_side + loc.location_in_blk.x; 
        block_idx = loc.location_in_grid.x * blocks->grid_side  + loc.location_in_grid.y; 
        //printf("after collapsing\n"); 
        //grd_print(NULL, blocks); 
        set_mask(my_masks, col_idx, row_idx, block_idx, blocks->states[idx]); 
        if(my_masks->safe_exit == 1)
        {
            printf("conflict while collapsing, safe exiting\n"); 
            return false; 
        }
        //print_masks(my_masks, blocks->block_side, blocks->grid_side);
        // Propagate
        // In the block
        //printf("Before propagate\n"); 
        //grd_print(NULL, blocks);
        
        if(propagate_all(blocks,
                      loc.location_in_grid.x,
                      loc.location_in_grid.y,
                      loc.location_in_blk.x,
                      loc.location_in_blk.y,
                      blocks->states[idx], my_masks) == false) {
                        
                        
                        return false; } 
        //grd_print(NULL, blocks); 
        //print_masks(my_masks, blocks->block_side, blocks->grid_side);

        iteration += 1;
        //if (iteration == 1) break; 
    }

    //print the last state 
    printf("last state: \n"); 
    grd_print(NULL, blocks); 

    // return D
    return true;
}
