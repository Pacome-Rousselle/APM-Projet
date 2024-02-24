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
/*MASKS EXP: 
each of them willbe starting as all the bits set to 1 
then wen a state becomes defined case it will put the related mask to zero 
*/


typedef struct{
    uint64_t* column_masks;  //gridside * blockside numbe rof elements they hold 
    uint64_t* row_masks; 
    uint64_t* block_masks; 
    //maybe at least this will be aligned :( 
    uint64_t safe_exit;  //if it is zero it means that mask setting or whatever we are in is successfull 
                            //if it is not zero, we are going to try a find a way to exit and return to the main 
                                //try new seed or jsut exit faiulure 
}masks; 

typedef masks *masks_ptr; 
/*had to change the signature of the load_save function to retrieve the masks to the main*/
typedef struct {
    masks* return_masks; 
    wfc_blocks* return_blocks;
    
}wfc_load_returns; 

typedef struct {
    const char *const data_file;
    const char *const output_folder;
    seeds_list *restrict seeds;
    const uint64_t parallel;
    bool (*solver)(wfc_blocks_ptr, uint64_t, masks*);
} wfc_args;

typedef struct {
    const char *const name;
    bool (*function)(wfc_blocks_ptr,  uint64_t,  masks*);
} wfc_solver;


