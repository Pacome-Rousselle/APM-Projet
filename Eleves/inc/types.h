#pragma once

#include <inttypes.h>
#include <stdbool.h>

/// Opaque type to store the seeds to try for the solving process. You may push to it and pop from
/// it. You may not try to index it manually or free this structure, it will be automatically freed
/// when no more items are present inside it.
typedef struct seeds_list seeds_list;

typedef struct {
    uint32_t x, y;
} vec2;

//ça casse alignement
typedef struct {
    vec2 location_in_blk; //64
    vec2 location_in_grid; //64
    uint8_t entropy;    //8
    uint8_t _1;         //8
    uint16_t _2;        //16
} entropy_location;

typedef struct {
    uint8_t block_side;
    uint8_t grid_side;

    uint8_t _1;
    uint8_t _2;
    uint32_t _3;

    uint64_t states[];
} wfc_blocks;

typedef wfc_blocks *wfc_blocks_ptr;

typedef struct {
    const char *const data_file;
    const char *const output_folder;
    seeds_list *restrict seeds;
    const uint64_t parallel;
    bool (*const solver)(wfc_blocks_ptr, uint64_t);
} wfc_args;

typedef struct {
    const char *const name;
    bool (*function)(wfc_blocks_ptr);
} wfc_solver;
