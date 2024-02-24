#define _GNU_SOURCE

#include "wfc.h"
#include "bitfield.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <inttypes.h> //to correctly print some values

/// With a block side of 8, we have blocks of 8*8 := 64, which is the number of bits in an uint64_t.
static const uint8_t BLOCK_SIDE_U64 = 8;

static void
trim(char *restrict str)
{
    unsigned long start = 0, end = strlen(str) - 1;

    while (isspace(str[start])) {
        start++;
    }

    while (end > start && isspace(str[end])) {
        end--;
    }

    if (start > 0 || end < (strlen(str) - 1)) {
        memmove(str, str + start, end - start + 1);
        str[end - start + 1] = '\0';
    }
}

static char *
next(char *restrict str, char sep)
{
    char *ret = strchr(str, sep);
    if (NULL == ret) {
        fprintf(stderr, "failed to find character '%c'\n", sep);
        exit(EXIT_FAILURE);
    }
    ret[0] = '\0';
    ret += 1;
    return ret;
}

static inline wfc_blocks *
safe_malloc(uint64_t blkcnt)
{
    uint64_t size   = sizeof(wfc_blocks) + sizeof(uint64_t) * blkcnt;
    wfc_blocks *ret = (wfc_blocks *)malloc(size);
    if (ret != NULL) {
        return ret;
    } else {
        fprintf(stderr, "failed to malloc %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
}

static inline uint32_t
to_u32(const char *string)
{
    char *end          = NULL;
    const long integer = strtol(string, &end, 10);
    if (integer < 0) {
        fprintf(stderr, "expected positive integer, got %ld\n", integer);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)integer;
}

static inline uint32_t
to_u64(const char *string)
{
    char *end               = NULL;
    const long long integer = strtoll(string, &end, 10);
    if (integer < 0) {
        fprintf(stderr, "expected positive integer, got %lld\n", integer);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)integer;
}

void
setBit(uint64_t *val, int bit, int value)
{
    if (value) {
        *val |= ((uint64_t)1 << bit); // Set the bit
    } else {
        *val &= ~((uint64_t)1 << bit); // Clear the bit
    }
}



//wfc_blocks_ptr
wfc_load_returns*
wfc_load(uint64_t seed, const char *path)
{
    srandom((uint32_t)seed);

    ssize_t read    = -1;
    char *line      = NULL;
    size_t len      = 0;
    wfc_blocks *ret = NULL;
    uint64_t blkcnt = 0;

    /*masks*/
    masks *my_masks = (masks *)malloc(sizeof(masks));
    if (my_masks == NULL) {
        printf("error allocating the masks bugffer\n");
        exit(EXIT_FAILURE);
    }
    
 

    FILE *restrict const f = fopen(path, "r");
    if (NULL == f) {
        fprintf(stderr, "failed to open `%s`: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if ((read = getline(&line, &len, f)) != -1) {
        const uint32_t block_side = to_u32(&line[1]);
        if (block_side > BLOCK_SIDE_U64) {
            fprintf(stderr, "invalid header of .dat file\n");
            exit(EXIT_FAILURE);
        }

        if (line[0] == 's') {
            blkcnt          = block_side * block_side;
            ret             = safe_malloc(blkcnt + wfc_control_states_count(1, block_side));
            ret->block_side = (uint8_t)block_side;
            ret->grid_side  = 1u;

            my_masks->column_masks = (uint64_t *)malloc(sizeof(uint64_t) * block_side);
            if (my_masks->column_masks == NULL) {
                printf("error allocating column masks\n");
                exit(EXIT_FAILURE);
            }
            my_masks->row_masks = (uint64_t *)malloc(sizeof(uint64_t) * block_side);
            if (my_masks->row_masks == NULL) {
                printf("error allocating row masks\n");
                exit(EXIT_FAILURE);
            }
               my_masks->block_masks = (uint64_t *)malloc(sizeof(uint64_t) * ret->grid_side* ret->grid_side);
            if (my_masks->row_masks == NULL) {
                printf("error allocating row masks\n");
                exit(EXIT_FAILURE);
            }
        } else if (line[0] == 'g') {
            blkcnt          = block_side * block_side;
            blkcnt          = blkcnt * blkcnt;
            ret             = safe_malloc(blkcnt + wfc_control_states_count(block_side, block_side));
            ret->block_side = (uint8_t)block_side;
            ret->grid_side  = (uint8_t)block_side;

            my_masks->column_masks = (uint64_t *)malloc(sizeof(uint64_t) * ret->block_side * ret->grid_side);
            if (my_masks->column_masks == NULL) {
                printf("error allocating column masks\n");
                exit(EXIT_FAILURE);
            }
            my_masks->row_masks = (uint64_t *)malloc(sizeof(uint64_t) * ret->block_side * ret->grid_side);
            if (my_masks->row_masks == NULL) {
                printf("error allocating row masks\n");
                exit(EXIT_FAILURE);
            }
            my_masks->block_masks = (uint64_t *)malloc(sizeof(uint64_t) * ret->grid_side* ret->grid_side);
            if (my_masks->row_masks == NULL) {
                printf("error allocating row masks\n");
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "invalid header of .dat file\n");
            exit(EXIT_FAILURE);
        }
    } else {
        fprintf(stderr, "invalid header of .dat file\n");
        exit(EXIT_FAILURE);
    }

    { ///////////here it is to give all the statees 1111111
        uint64_t mask       = 0;
        const uint64_t base = wfc_control_states_count(ret->grid_side, ret->block_side);
        for (uint8_t i = 0; i < ret->block_side * ret->block_side; i += 1) {
            mask = bitfield_set(mask, i); ///over writing the same value multiple times so taht every one will 1111s in the end
            //printf("mask %lu \n", mask);
        }
        ret->states[0] = seed; //unnecessary it will be over written
        for (uint64_t i = 0; i < blkcnt + base; i += 1) {
            ret->states[i] = mask;
        }

        for (uint64_t i = 0; i < ret->grid_side * ret->block_side; i++) {
            my_masks->column_masks[i] = mask;
            my_masks->row_masks[i]    = mask; 
        }

         for (uint64_t i = 0; i < ret->grid_side * ret->grid_side; i++) {
            my_masks->block_masks[i] = mask;
            
        }
    }

    while ((read = getline(&line, &len, f)) != -1) {
        trim(line);

        char *str_gx      = line;
        char *str_gy      = next(str_gx, ',');
        char *str_x       = next(str_gy, ',');
        char *str_y       = next(str_x, ',');
        char *str_state   = next(str_y, '=');
        const uint32_t gx = to_u32(str_gx), gy = to_u32(str_gy), x = to_u32(str_x),
                       y = to_u32(str_y);

        if (gx >= ret->grid_side || gy >= ret->grid_side) {
            fprintf(stderr, "invalid grid coordinates (%u, %u)\n", gx, gy);
            exit(EXIT_FAILURE);
        } else if (x >= ret->block_side || y >= ret->block_side) {
            fprintf(stderr, "invalid block coordinates (%u, %u) in grid (%u, %u)\n", x, y, gx, gy);
            exit(EXIT_FAILURE);
        }

        const uint64_t collapsed = to_u64(str_state);
        //printf("locations:gx %u  gy %u x %u y %u value: %lu \n",gx, gy, x, y, collapsed);

        uint64_t val = 0;
        val          = convertToBits(collapsed);
        /*since this is the first iteration, every incoming value, if they are correctly given, 
        will set to 0 the location OF THE NUMBER to zero; indicating that the value is not usable anymore*/
        /*we have to know on which column and on which row we are */
        uint32_t col_idx = gy * ret->block_side + y; 
        uint32_t row_idx = gx * ret->block_side + x; 
        uint32_t block_idx = gx * ret->grid_side + gy; 

       
         my_masks->safe_exit= 0; 
        set_mask(my_masks, col_idx, row_idx, block_idx, val); 
        if ( (my_masks->safe_exit) == 1)
        {
            printf("couldnt set the mask \n"); 
            printf("an invalid type of configuration is given, exiting\n"); 
            exit(EXIT_FAILURE);  //since we are in the entry and there had been no propogations we ar eexiting 
        }
        print_masks(my_masks, ret->block_side, ret->grid_side); 


        *blk_at(ret, gx, gy, x, y) = val;
        //printf("in load_wfc function\n");
        grd_print(NULL, ret);
        blk_propagate(ret, gx, gy, val, NULL, NULL, my_masks);
        ////printf("blk_propagate\n");
        //grd_print(NULL, ret);
        grd_propagate_column(ret, gx, gy, x, y, val, NULL, NULL, my_masks);
        ////printf("calculate index %u\n ", get_thread_glob_idx(ret, gx, gy, x, y));
        ////printf("grd_propogate_col\n");
        ////grd_print(NULL, ret);
        grd_propagate_row(ret, gx, gy, x, y, val, NULL, NULL, my_masks);
        ////printf("grd_propogate_row\n");
        ////grd_print(NULL, ret);
//
        //if (grd_check_error_in_column(ret, x, gx) == false) {
        //    fprintf(stderr, "wrong propagation in block (%u, %u) from (%u, %u)\n", gx, gy, x, y);
        //    exit(EXIT_FAILURE);
        //}
    }

    printf("load_wfc is finished\n");
    //grd_print(NULL, ret); 
    free(line);
    fclose(f);
    //return ret;
       /*return of this function*/
    wfc_load_returns* returns = (wfc_load_returns*) malloc(sizeof(wfc_load_returns)); 
    returns->return_blocks = ret; 
    returns->return_masks = my_masks; 



    return returns; 
}

void
wfc_save_into(const wfc_blocks_ptr blocks, const char data[], const char folder[])
{
    char destination[1024] = { 0 };
    const size_t data_len  = strlen(data);
    const char *file_name  = &data[data_len - 1];
    while (file_name != data && file_name[0] != '/') {
        file_name -= 1;
    }
    const char *file_end = strchr(file_name, '.');
    long length          = (file_end - file_name);
    if (length >= 1024) {
        length = 1023;
    } else if (length < 0) {
        length = 0;
    }

    const size_t folder_len = strlen(folder);
    if (folder[folder_len - 1] == '/' && file_name[0] == '/') {
        snprintf(destination, 1023, "%.*s%.*s.%lu.save", (int)(folder_len - 1), folder, (int)length,
                 file_name, blocks->states[0]);
    } else if ((folder[folder_len - 1] == '/' && file_name[0] != '/') ||
               (folder[folder_len - 1] != '/' && file_name[0] == '/')) {
        snprintf(destination, 1023, "%s%.*s.%lu.save", folder, (int)length, file_name,
                 blocks->states[0]);
    } else {
        snprintf(destination, 1023, "%s/%.*s.%lu.save", folder, (int)length, file_name,
                 blocks->states[0]);
    }
    fprintf(stdout, "save result to file: %s\n", destination);

    FILE *restrict f = fopen(destination, "w");
    if (NULL == f) {
        fprintf(stderr, "failed to open file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (fprintf(f, "grid:  %hhu\n", blocks->grid_side) < 0) {
        fprintf(stderr, "failed to write: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    if (fprintf(f, "block: %hhu\n", blocks->block_side) < 0) {
        fprintf(stderr, "failed to write: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    // const uint64_t starts = wfc_control_states_count(blocks->grid_side, blocks->block_side),
    const uint64_t ends = blocks->grid_side * blocks->grid_side * blocks->block_side *
                          blocks->block_side;
    uint8_t gs = blocks->grid_side;
    uint8_t bs = blocks->block_side;

    for (uint32_t ii = 0; ii < gs; ii++)
        for (uint32_t jj = 0; jj < bs; jj++) {
            for (uint32_t i = 0; i < gs; i++)
                for (uint32_t j = 0; j < bs; j++) {
                    int pow    = 1;
                    int number = *blk_at(blocks, ii, i, jj, j);
                    while (number != 1) {
                        number /= 2;
                        if (number < 1)
                            fprintf(stderr, "bad result in the grid: %s\n", strerror(errno));
                        pow++;
                    }
                    if (fprintf(f, "%d ", pow) < 0) {
                        fprintf(stderr, "failed to write: %s\n", strerror(errno));
                        exit(EXIT_FAILURE);
                    }
                }
            fprintf(f, "\n");
        }

    fprintf(stdout, "saved successfully %lu states\n", ends);
    fclose(f);
}
