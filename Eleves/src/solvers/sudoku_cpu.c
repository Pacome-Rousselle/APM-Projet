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
    int changed = 0; 
/*printing the current stuation in the beginning*/

printf("grid side: %u\n", blocks->grid_side); 
printf("block side: %u\n", blocks->block_side); 
// Propagate the seed
blk_propagate(blocks,
              0,
              0,
              blocks->states[0]);
              
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
        //  D[x,y] = collapse_state (D[x,y]) -> How do I get this
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
        //// Propagate
        //// In the block
        printf("before propagate\n"); 
        grd_print(NULL, blocks);
        // BLOCK
        blk_propagate(blocks,
                      loc.location_in_grid.x,
                      loc.location_in_grid.y,
                      blocks->states[idx]);
        printf("after block\n"); 
        grd_print(NULL, blocks);

        if(grd_check_error_in_blk(blocks, loc.location_in_grid.x, 
                                  loc.location_in_grid.y, 
                                  loc.location_in_blk.x,
                                  loc.location_in_blk.y) == false ){
            printf("encountered another state same value in same block\n"); 
            return false; 
        };  
        // COL
        grd_propagate_column(blocks, loc.location_in_grid.x, loc.location_in_grid.y, 
                            loc.location_in_blk.x, loc.location_in_blk.y, blocks->states[idx]);
        printf("after grid col\n"); 
        grd_print(NULL, blocks);

        if(grd_check_error_in_column(blocks, loc.location_in_grid.y,loc.location_in_blk.y) == false ){
            printf("encountered another state same value in same column\n"); 
            return false; 
        };  

        // ROW
        grd_propagate_row(blocks, loc.location_in_grid.x, loc.location_in_grid.y, 
                            loc.location_in_blk.x, loc.location_in_blk.y, blocks->states[idx]);
        
        printf("after grid row \n"); 
        grd_print(NULL, blocks);

        if(grd_check_error_in_row(blocks, loc.location_in_grid.x,loc.location_in_blk.x) == false ){
            printf("encountered another state same value in same row\n"); 
            return false; 
        };       

        iteration += 1;
        // if (iteration == 5 ){
        //     printf("iteration 5\n"); 
        //     break;
        // }
 
         //}
    }
    //print the last state 
    printf("last state: \n"); 
    grd_print(NULL, blocks); 

    // return D
    return true;
}

