#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

bool
solve_cpu(wfc_blocks_ptr blocks)
{
    uint64_t iteration  = 0;
    // Set seed
    const uint64_t seed = blocks->states[0];
    struct {
        uint32_t gy, x, y, _1;
        uint64_t state;
    } row_changes[blocks->grid_side];

    entropy_location loc, test;
    loc.entropy = UINT8_MAX;
    forever {
        //while find_min_entropy
        for (int gx = 0; gx < blocks->grid_side; gx++)
            for (int gy = 0; gy < blocks->grid_side; gy++)
            {
                test = blk_min_entropy(blocks,gx,gy);
                if(test.entropy < loc.entropy)
                    loc = test;
            }
        
        //  assert ( entropy (D[x,y]) != 0)
        if (loc.entropy == 0)
            return false;

        // Collapse
        //  D[x,y] = collapse_state (D[x,y]) -> How do I get this
        int idx = get_thread_glob_idx(blocks,
                                      loc.location_in_grid.x,
                                      loc.location_in_grid.y,
                                      loc.location_in_blk.x,
                                      loc.location_in_blk.y);

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

        // Get the row and column of the state that collapsed

        // grd_propagate_row(blocks,1,0,blocks->states[idx],);
        // grd_propagate_column(blocks,1,0,blocks->states[idx]);

        // Check Error
        //  if not update_domaine (D,x,y):
        //       return " propagation failed "
        // bool changed = false;
        

        iteration += 1;
        // if (!changed)
        //     break;
    }
    // return D
    return true;
}
