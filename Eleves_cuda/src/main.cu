//#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <float.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>

int
main(int argc, char **argv)
{
    omp_set_dynamic(false);

    wfc_args args             = wfc_parse_args(argc, argv);
    
    //const wfc_blocks_ptr init = wfc_load(0, args.data_file);
    const wfc_load_returns* load = wfc_load(0, args.data_file);
    const wfc_blocks_ptr init  = load->return_blocks; 
    const masks_ptr my_masks_init = load->return_masks; //initial entry states' masks 

    masks* fresh_mask = (masks *)malloc(sizeof(masks)); 
    masks* my_mask = (masks *)malloc(sizeof(masks)); 

    fresh_mask->column_masks = (uint64_t *)malloc(sizeof(uint64_t) * init->block_side * init->grid_side);
    fresh_mask->row_masks = (uint64_t *)malloc(sizeof(uint64_t) * init->block_side * init->grid_side);
    fresh_mask->block_masks = (uint64_t *)malloc(sizeof(uint64_t) * init->grid_side* init->grid_side);
     
    my_mask->column_masks = (uint64_t *)malloc(sizeof(uint64_t) * init->block_side * init->grid_side);
    my_mask->row_masks = (uint64_t *)malloc(sizeof(uint64_t) * init->block_side * init->grid_side);
    my_mask->block_masks = (uint64_t *)malloc(sizeof(uint64_t) * init->grid_side* init->grid_side);

    for(int i = 0; i <init->block_side * init->grid_side; i++ )
    {
        fresh_mask->column_masks[i]= load->return_masks->column_masks[i]; 
        fresh_mask->row_masks[i]= load->return_masks->row_masks[i]; 

    }
    for(int i = 0; i < init->grid_side* init->grid_side; i++){
        fresh_mask->block_masks[i]= load->return_masks->block_masks[i]; 

    }
    fresh_mask->safe_exit = 0; 

    int quit                 = 0;
    uint64_t iterations      = 0;
    wfc_blocks_ptr blocks    = NULL;

    //masks_ptr my_masks = NULL; 

    pthread_mutex_t seed_mtx = PTHREAD_MUTEX_INITIALIZER;

    int *volatile const quit_ptr           = &quit;
    uint64_t *volatile const iterations_ptr = &iterations;

    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start            = omp_get_wtime();

    while (!*quit_ptr) {
        pthread_mutex_lock(&seed_mtx);
        uint64_t next_seed       = 0;
        const bool has_next_seed = try_next_seed(&args.seeds, &next_seed);
        //printf("Seed : %d\n",next_seed);
        pthread_mutex_unlock(&seed_mtx);

        if (!has_next_seed) {
            __atomic_fetch_or(quit_ptr, (int)1, __ATOMIC_SEQ_CST);
            fprintf(stderr, "no more seed to try\n");
            break;
        }
       
        wfc_clone_into(&blocks, next_seed, init);
         



    for(int i = 0; i <init->block_side * init->grid_side; i++ )
    {
         my_mask->column_masks[i]=fresh_mask->column_masks[i] ;
         my_mask->row_masks   [i]=fresh_mask->row_masks[i]    ; 

    }
    for(int i = 0; i < init->grid_side* init->grid_side; i++){
         my_mask->block_masks[i]=fresh_mask->block_masks[i]    ; 
        

    }
    my_mask->safe_exit = 0; 
    //printf("my_masks\n"); 
    //print_masks(my_mask, init->block_side , init->grid_side); 


        const bool solved = args.solver(blocks,next_seed, my_mask);
        //printf("Atomic 1\n");
           
        __atomic_add_fetch(iterations_ptr, 1, __ATOMIC_SEQ_CST);
 
        if (solved && args.output_folder != NULL) { //change it in here 
            //printf("Atomic 2\n");
            __atomic_fetch_or(quit_ptr, (int)1, __ATOMIC_SEQ_CST);
            fputc('\n', stdout);
            wfc_save_into(blocks, args.data_file, args.output_folder);
        }

        else if (solved) {
            //printf("Atomic 3\n");
            __atomic_fetch_or(quit_ptr, (int)1, __ATOMIC_SEQ_CST);
            fputs("\n success with result:\n", stdout);
            abort();
        }

        else if (!*quit_ptr) {
            //printf("Atomic 4\n");
            fprintf(stdout, "\r%.2f%% -> %.2fs\n",
                    ((double)(*iterations_ptr) / (double)(max_iterations)) * 100.0,
                    omp_get_wtime() - start);
        }
    }

    return 0;
}

/*int
main(int argc, char **argv)
{
    omp_set_dynamic(false);

    wfc_args args             = wfc_parse_args(argc, argv);
    const wfc_blocks_ptr init = wfc_load(0, args.data_file);

   // bool quit                = false;
    int quit                 = 0;
    uint64_t iterations      = 0;
    wfc_blocks_ptr blocks    = NULL;
    pthread_mutex_t seed_mtx = PTHREAD_MUTEX_INITIALIZER;

    int *volatile const quit_ptr           = &quit;
    uint64_t *volatile const iterations_ptr = &iterations;

    const uint64_t max_iterations = count_seeds(args.seeds);
    printf("count seeds : %lu\n ", max_iterations); 
    const double start            = omp_get_wtime();

    while (!*quit_ptr) {
        pthread_mutex_lock(&seed_mtx);
        uint64_t next_seed       = 0;
        const bool has_next_seed = try_next_seed(&args.seeds, &next_seed);
        printf("next seed : %lu\n", next_seed); 
        pthread_mutex_unlock(&seed_mtx);

        if (!has_next_seed) {
            __atomic_fetch_or(quit_ptr, (int)1, __ATOMIC_SEQ_CST);
            fprintf(stderr, "no more seed to try\n");
            break;
        }

        wfc_clone_into(&blocks, next_seed, init);
        const bool solved = args.solver(blocks);
        
        __atomic_add_fetch(iterations_ptr, 1, __ATOMIC_SEQ_CST);

        if (solved && args.output_folder != NULL) {
            __atomic_fetch_or(quit_ptr, (int)1, __ATOMIC_SEQ_CST);
            fputc('\n', stdout);
            wfc_save_into(blocks, args.data_file, args.output_folder);
        }

        else if (solved) {
            __atomic_fetch_or(quit_ptr, (int)1, __ATOMIC_SEQ_CST);
            fputs("\nsuccess with result:\n", stdout);
            abort();
        }

        else if (!*quit_ptr) {
            printf("\n main: \n");
            fprintf(stdout, "\r%.2f%% -> %.2fs",
                    ((double)(*iterations_ptr) / (double)(max_iterations)) * 100.0,
                    omp_get_wtime() - start);
        }
    }

    return 0;
}*/
