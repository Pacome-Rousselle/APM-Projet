#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

bool
solve_cpu(wfc_blocks_ptr blocks, uint64_t seed)
{
    printf("inside solver cpu \n"); 
    uint64_t iteration  = 0;
    // Set seed
    // const uint64_t seed = blocks->states[0];
    // struct {
    //     uint32_t gy, x, y, _1;
    //     uint64_t state;
    // } row_changes[blocks->grid_side];

    entropy_location loc, test;
    int changed = 0; 
    /*printing the current stuation in the beginning*/

    printf("grid side: %u\n", blocks->grid_side); 
    printf("block side: %u\n", blocks->block_side); 
    
    grd_print(NULL, blocks);
    forever {
        printf("Itération %d\n",iteration);
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
        
        //  assert ( entropy (D[x,y]) != 0)
        printf("loc.entropy is %zu\n", loc.entropy); 
        if (loc.entropy == 1)
        {
        printf("inside loc.entropy = 1\n"); 
            return false;
        }

        // Collapse
        int idx = get_thread_glob_idx(blocks,
                                      loc.location_in_grid.x,
                                      loc.location_in_grid.y,
                                      loc.location_in_blk.x,
                                      loc.location_in_blk.y);
        printf("idx: %d\n", idx); 
        blocks->states[idx] = entropy_collapse_state(blocks->states[idx],
                                                    loc.location_in_grid.x,
                                                    loc.location_in_grid.y,
                                                    loc.location_in_blk.x,
                                                    loc.location_in_blk.y,
                                                    seed,
                                                    iteration);
        // Propagate
        // In the block
        printf("Before propagate\n"); 
        grd_print(NULL, blocks);
        
        if(propagate_all(blocks,
                      loc.location_in_grid.x,
                      loc.location_in_grid.y,
                      loc.location_in_blk.x,
                      loc.location_in_blk.y,
                      blocks->states[idx]) == false) return false;  


        iteration += 1;
    }

    //print the last state 
    printf("last state: \n"); 
    grd_print(NULL, blocks); 

    // return D
    return true;
}

