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

    entropy_location loc;
    forever {
        //while find_min_entropy
        loc = blk_min_entropy(blocks,blocks->grid_side,blocks->grid_side);
        //  assert ( entropy (D[x,y]) != 0)
        if (loc.entropy == 0)
            return false;
        // Collapse
        //  D[x,y] = collapse_state (D[x,y]) -> How do I get this
        int idx = get_thread_glob_idx
        (blocks,blocks->grid_side,blocks->grid_side,loc.location_in_blk.x,loc.location_in_blk.y);
        
        blocks->states[idx];
        // Propagate
        //  if not update_domaine (D,x,y):
        //       return " propagation failed "
        // return D
        bool changed = false;
        // 3. Check Error

        iteration += 1;
        if (!changed)
            break;
    }

    return true;
}
