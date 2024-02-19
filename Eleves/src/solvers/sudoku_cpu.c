#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

bool
solve_cpu(wfc_blocks_ptr blocks)
{
    printf("inside solver cpu \n"); 
    uint64_t iteration  = 0;
    // Set seed
    const uint64_t seed = blocks->states[0];
    struct {
        uint32_t gy, x, y, _1;
        uint64_t state;
    } row_changes[blocks->grid_side];

    entropy_location loc, test;
    loc.entropy = UINT8_MAX;
    int changed = 0; 
/*printing the current stuation in the beginning*/

printf("grid side: %u\n ", blocks->grid_side); 
printf("block side: %u\n ", blocks->block_side); 

grd_print(NULL, blocks); 

    forever {
        //while find_min_entropy in grid 
         
        for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++)
            {
                test = blk_min_entropy(blocks,gx,gy);
                if(test.entropy < loc.entropy)
                    loc = test;
            }
        
        //  assert ( entropy (D[x,y]) != 0)
        printf("loc.entropy is %zu\n", loc.entropy); 
        //if (loc.entropy == 1)
        //{
        //printf("inside loc.entropy = 1\n"); 
        //    return false;
//
        //}

        // Collapse
        //  D[x,y] = collapse_state (D[x,y]) -> How do I get this
        int idx = get_thread_glob_idx(blocks,
                                      loc.location_in_grid.x,
                                      loc.location_in_grid.y,
                                      loc.location_in_blk.x,
                                      loc.location_in_blk.y);
        printf("idx: %d\n", idx); 
        if (entropy_compute(blocks->states[idx]) != 1)
        {
            blocks->states[idx] = entropy_collapse_state(blocks->states[idx],
                                                    loc.location_in_grid.x,
                                                    loc.location_in_grid.y,
                                                    loc.location_in_blk.x,
                                                    loc.location_in_blk.y,
                                                    seed,
                                                    iteration);
        // Propagate
        // In the block
        blk_propagate(blocks,
                      loc.location_in_grid.x,
                      loc.location_in_grid.y,
                      blocks->states[idx]);

        grd_propagate_row(blocks,loc.location_in_grid.x,
                          loc.location_in_grid.y,
                          loc.location_in_blk.x,
                          loc.location_in_blk.y,
                          blocks->states[idx]);

        grd_propagate_column(blocks,loc.location_in_grid.x,
                          loc.location_in_grid.y,
                          loc.location_in_blk.x,
                          loc.location_in_blk.y,
                          blocks->states[idx]);                 
        // grd_propagate_column(blocks,1,0,blocks->states[idx]);

        // Check Error
        //  if not update_domaine (D,x,y):
        //       return " propagation failed "
        // bool changed = false;
        

        iteration += 1;
        if (iteration == 1 ){
            printf("iteration 5\n"); 
            break;
        }
        // if (!changed)
        // {
        //printf("inside !changed\n"); 
        //     break;
//
        // }
    }
    // return D
    return true;
}
}
