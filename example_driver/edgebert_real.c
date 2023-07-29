/* Copyright (c) 2011-2021 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */

#include "edgebert_real.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <esp_accelerator.h>
#include <esp_probe.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

// Platform-Level Interrupt Controller (PLIC) (base address)
#define PLIC_ADDR 0x6c000000
#define PLIC_IP_OFFSET 0x1000       // Interrupt Pending
#define PLIC_INTACK_OFFSET 0x200004 // Interrupt Acknowledge
#define EDGEBERT_IRQ 5

#define MAX_PRINTED_ERRORS 10
#define HU_EDGEBERT 0x104
#define DEV_NAME "hu,hu_edgebert"

typedef char token_t;
typedef char native_t;
struct mat {
    token_t *values;
    token_t *mask;
    int bias;
};
const int bits_in_bytes = 8;

// Get direct memory access (DMA) per beat
static unsigned DMA_WORD_PER_BEAT(unsigned _st) {
    return (sizeof(void *) / _st);
}

// Config
// Constants
// Memory size
static unsigned mem_size;
// Size of mask buffer (in bytes)
const static unsigned mask_buffer_size = 8192;
// Size of data buffer (in bytes)
const static unsigned input_buffer_size = 65536;
// Size of aux buffer (in bytes)
const static unsigned aux_buffer_size = 4096;

// EdgeBert constants
const static int vector_size = 16;
const static int adf_accum_bias = 2;
const static int accum_right_shift_base = 4;

// Used to calculate time (get current counter)
static inline uint64_t get_counter() {
    uint64_t counter;
    asm volatile (
        "li t0, 0;"
        "csrr t0, mcycle;"
        "mv %0, t0"
        : "=r" ( counter )
        :
        : "t0"
    );
    return counter;
}


// CPU functions for profiling
// Data
// Input initialization
// static void CPU_EdgeBert_init_buf_attention(int *input_ids, int *we_query, int *we_key, int *we_val) {
//     #include "data/attention/input_ids.h"     // 128 * 768
//     #include "data/attention/we_query.h"      // 768 * 64
//     #include "data/attention/we_key.h"        // 768 * 64
//     #include "data/attention/we_val.h"        // 768 * 64
// }

// static void CPU_EdgeBert_init_buf_pooler(int *we_mat1) {
//     #include "data/pooler/we_mat1.h"          // 64 * 64
// }

// static void CPU_EdgeBert_init_buf_highway(int *we_mat1) {
//     #include "data/highway/we_mat1.h"         // 64 * 2
// }

// static void CPU_EdgeBert_init_buf_processing(int *we_mat1) {
//     #include "data/processing/we_mat1.h"      // 768 * 768
// }

// static void CPU_EdgeBert_init_buf_ffn(int *we_mat1, int *we_mat2) {
//     #include "data/ffn/we_mat1.h"             // 768 * 3072
//     #include "data/ffn/we_mat2.h"             // 3072 * 768
// }

// Helper functions
// Min of two integers
int min(int a, int b) {
    return (a < b)? a : b;
}

// Transpose for integer
void CPU_transpose_int(int *array, int m, int n) {
    int new_array[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Index in the original matrix
            int index1 = i * n + j;
            // Index in the transpose matrix
            int index2 = j * m + i;
            // Swap
            new_array[index2] = array[index1];
        }
    }

    // Replace in place
    for (int i = 0; i < m * n; i++) {
        array[i] = new_array[i];
    }
}

// Softmax over array of size size (in-place)
// void CPU_softmax(float *input, size_t size) {
//     // Scale down to prevent overflow
//     int i;
//     float m, sum, constant;

//     // Find max of array
//     m = -INFINITY;
//     for (i = 0; i < size; i++) {
//         if (m < input[i]) {
//             m = input[i];
//         }
//     }

//     // Subtract off largets element and exp
//     sum = 0.0;
//     for (i = 0; i < size; i++) {
//         sum += exp(input[i] - m);
//     }

//     // Subtract off max and put sum in denominator
//     constant = m + log(sum);
//     for (i = 0; i < size; i++) {
//         input[i] = exp(input[i] - constant);
//     }
// }

// Multiply a N0 x M_mat matrix at a and M_mat x N1 matrix at b and store in d
void CPU_multiply(int *a, int *b, int N0, int M_mat, int N1, int *d) {
    for (int i = 0; i < N0; i++) {
        for (int j = 0; j < N1; j++) {
            // Get d[i][j] by getting dot product of row i and column j
            int sum = 0;
            for (int k = 0; k < M_mat; k++) {
                sum = sum + a[i * M_mat + k] * b[k * N1 + j];
            }
            d[i * N1 + j] = sum;
        }
    }
}

// Calculate entropy
// float *CPU_entropy(float *input, int m, int n) {
//     float e[m * n];
//     float e_x[m * n];
//     // Apply exp to each entry
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             e[i * n + j] = exp(input[i * n + j]);
//             e_x[i * n + j] = input[i * n + j] * exp(input[i * n + j]);;
//         }
//     }

//     // Sum over rows
//     float a[m];
//     float b[m];
//     for (int i = 0; i < m; i++) {
//         float sum_a = 0.0;
//         float sum_b = 0.0;
//         for (int j = 0; j < n; j++) {
//             sum_a += e[i * n + j];
//             sum_b += e_x[i * n + j];
//         }
//         a[i] = sum_a;
//         b[i] = sum_b;
//     }

//     // Calculate entropy
//     float *out = aligned_malloc(m * sizeof(float));
//     for (int i = 0; i < m; i++) {
//         out[i] = log(a[i]) - b[i] / a[i];
//     }
//     return out;
// }


// CPU functions for EdgeBert
// Transpose a matrix of chars at array with original size m x n (in-place)
void CPU_transpose(token_t *array, int m, int n) {
    token_t new_array[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Index in the original matrix
            int index1 = i * n + j;
            // Index in the transpose matrix
            int index2 = j * m + i;
            // Swap
            new_array[index2] = array[index1];
        }
    }

    // Replace in place
    for (int i = 0; i < m * n; i++) {
        array[i] = new_array[i];
    }
}

// Decode a m x n matrix by returning full array
token_t *CPU_decode_matrix(token_t *arr, int m, int n, token_t *mask) {
    // Allocate space for decoded matrix
    token_t *out = aligned_malloc(m * n);
    int arr_idx = 0;

    // Iterate over entries
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int mask_idx = (i * n + j) / 8;
            int offset = (i * n + j) % 8;

            // Add entry if mask is 1
            if (((mask[mask_idx] >> offset) & 1) == 1) {
                out[i * n + j] = arr[arr_idx];
                arr_idx++;
            } else {
                out[i * n + j] = 0;
            }
        }
    }
    return out;
}

// Encode a m x n matrix by filling mask and returning condensed array
token_t *CPU_encode_matrix(token_t *array, int m, int n, token_t *mask) {
    // Clear out mask
    memset(mask, 0, m * n);
    int non_zero = 0;

    // Find indices that are non-zero and set mask
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int mask_idx = (i * n + j) / 8;
            int offset = (i * n + j) % 8;

            if (array[i * n + j] != 0) {
                mask[mask_idx] += (1 << offset);
                non_zero++;
            }
        }
    }

    // Return condensed array
    token_t *out = aligned_malloc(non_zero);
    int idx = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (array[i * n + j] != 0) {
                out[idx] = array[i * n + j];
                idx++;
            }
        }
    }
    return out;
}

// Transformer profiling
// Pooler
static int *CPU_EdgeBert_pooler(
    int *attention_heads,
    int input_m,
    int hidden_size
) {
    // Weight matrix (hidden_size * hidden_size)
    int *we_mat1;
    int *output;

    // Allocate space for matrices
    we_mat1 = aligned_malloc(hidden_size * hidden_size * sizeof(int));
    output = aligned_malloc(input_m * hidden_size * sizeof(int));

    // Fill with data
    // CPU_EdgeBert_init_buf_pooler(we_mat1);
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        we_mat1[i] = 10;
    }

    // Matrix multiplication configurations
    int N0;
    int N1;
    int M_mat;
    N0 = input_m; M_mat = hidden_size; N1 = hidden_size;

    // Query multiplication
    CPU_multiply(we_mat1, attention_heads, N0, M_mat, N1, output);

    // Activation?

    // Get hidden states for first token
    int *out;
    out = aligned_malloc(hidden_size * sizeof(int));
    memcpy(out, output, hidden_size * sizeof(int));

    // Free allocated space
    aligned_free(we_mat1);
    aligned_free(output);

    return out;
}

// Highway Exit
static int *CPU_EdgeBert_highway_exit(
    int *attention_heads,
    int input_m,
    int hidden_size,
    int num_labels
) {
    int *pooler_output = CPU_EdgeBert_pooler(attention_heads, input_m, hidden_size);
    int *we_mat1;
    int *we_mat2;
    int *output1;
    int *output2;

    we_mat1 = aligned_malloc(num_labels * hidden_size * sizeof(int));
    output1 = aligned_malloc(num_labels * sizeof(int));

    // Fill with data
    // CPU_EdgeBert_init_buf_highway(we_mat1);
    for (int i = 0; i < num_labels * hidden_size; i++) {
        we_mat1[i] = 4;
    }

    // Mutliply pooler output by first matrix
    int N0;
    int N1;
    int M_mat;
    N0 = num_labels; M_mat = hidden_size; N1 = 1;
    CPU_multiply(we_mat1, pooler_output, N0, M_mat, N1, output1);
    aligned_free(pooler_output);

    // Entropy?
    // float entropy = CPU_entropy(output2, he_layer2, 1);
    // float threshold = 0.0;
    // if (entropy < threshold) {
    //     return output2;
    // }

    // Free allocated space
    aligned_free(we_mat1);

    return output1;
}

// Attention head
static int *CPU_EdgeBert_attention(
    int input_m,
    int input_n,
    int hidden_size
) {
    printf("STARTing Attention Head in CPU...\n");

    // Initialize inputs, weights, and outputs
    int *input_ids;
    int *we_query;
    int *we_key;
    int *we_val;
    int *output1;
    int *output2;
    int *output3;

    // Allocate space for matrices
    input_ids = aligned_malloc(input_m * input_n * sizeof(int));
    we_query = aligned_malloc(input_n * hidden_size * sizeof(int));
    we_key = aligned_malloc(input_n * hidden_size * sizeof(int));
    we_val = aligned_malloc(input_n * hidden_size * sizeof(int));
    output1 = aligned_malloc(input_m * hidden_size * sizeof(int));
    output2 = aligned_malloc(input_m * hidden_size * sizeof(int));
    output3 = aligned_malloc(input_m * hidden_size * sizeof(int));

    // Load in data
    // CPU_EdgeBert_init_buf_attention(input_ids, we_query, we_key, we_val);
    for (int i = 0; i < input_m * input_n; i++) {
        input_ids[i] = 12;
    }
    for (int i = 0; i < input_n * hidden_size; i++) {
        we_query[i] = 24;
        we_key[i] = -5;
        we_val[i] = 126;
    }

    // Matrix multiplication configurations
    int N0;
    int N1;
    int M_mat;
    N0 = input_m; M_mat = input_n; N1 = hidden_size;

    // Query multiplication
    CPU_multiply(input_ids, we_query, N0, M_mat, N1, output1);
    // Key multiplication
    CPU_multiply(input_ids, we_key, N0, M_mat, N1, output2);
    // Value multiplication
    CPU_multiply(input_ids, we_val, N0, M_mat, N1, output3);
    // Transpose key output
    CPU_transpose_int(output2, N0, N1);

    // Query output multiplied by transpose of key output
    N0 = input_m; M_mat = hidden_size; N1 = input_m;
    int *output4;
    output4 = aligned_malloc(input_m * input_m * sizeof(int));
    CPU_multiply(output1, output2, N0, M_mat, N1, output4);

    // Softmax?

    // Attention Span Mask?

    // Multiply value output
    N0 = input_m; M_mat = input_m; N1 = hidden_size;
    int *output5;
    output5 = aligned_malloc(input_m * hidden_size * sizeof(int));
    CPU_multiply(output4, output3, N0, M_mat, N1, output5);

    // Free allocated space
    aligned_free(input_ids);
    aligned_free(we_query);
    aligned_free(we_key);
    aligned_free(we_val);
    aligned_free(output1);
    aligned_free(output2);
    aligned_free(output3);
    aligned_free(output4);

    printf("FINISHing Attention Head in CPU...\n");
    return output5;
}

static int *CPU_EdgeBert_attention_heads(
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels
) {
    printf("STARTing CPU %d Attention Heads Computation...\n", num_heads);

    // Initialize outputs
    int *attention_heads;
    attention_heads = aligned_malloc(128 * 768 * sizeof(int));

    // Profile
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    // Run attention heads
    for (int i = 0; i < 12; i++) {
        // Get time for each head
        count1 = get_counter();
        int *attention_output = CPU_EdgeBert_attention(
            input_m,
            input_n,
            hidden_size
        );
        count2 = get_counter();
        exe_cycle = count2 - count1;
        printf("...Attention Head %d takes %"PRIu64" clock cycles...\n", i + 1, exe_cycle);
        total_exe_cycle = total_exe_cycle + exe_cycle;

        // Fill output with dummy data
        for (int l = 0; l < input_m; l++) {
            for (int k = 0; k < hidden_size; k++) {
                attention_heads[l * hidden_size * num_heads + i * hidden_size + k] = attention_output[l * hidden_size + k];
            }
        }

        // Highway exit
        int *highway_exit = CPU_EdgeBert_highway_exit(attention_output, input_m, hidden_size, num_labels);
        aligned_free(attention_output);
        aligned_free(highway_exit);

        // if (highway_exit) {
        //     printf("FINISHing CPU 12 Attention Heads Computation EARLY...\n");
        //     printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
        //     return highway_exit;
        // }
    }

    printf("FINISHing CPU 12 Attention Heads Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
    return attention_heads;
}

static int *CPU_EdgeBert_processing(
    int *attention_heads,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size
) {
    printf("STARTing CPU Attention Heads Processing...\n");
    // Profile
    uint64_t count1;
    uint64_t count2;
    count1 = get_counter();

    // Initialize weights for processing
    int *we_mat1;
    int *attention_heads_out;

    we_mat1 = aligned_malloc(hidden_size * num_heads * input_n * sizeof(int));
    attention_heads = aligned_malloc(input_m * input_n * sizeof(int));

    // Fill matrix with data
    // CPU_EdgeBert_init_buf_processing(we_mat1);
    for (int i = 0; i < hidden_size * num_heads * input_n; i++) {
        we_mat1[i] = -1;
    }

    // Multiply output by matrix
    int N0;
    int N1;
    int M_mat;
    N0 = input_m; M_mat = hidden_size * num_heads; N1 = input_n;
    CPU_multiply(attention_heads, we_mat1, N0, M_mat, N1, attention_heads_out);
    aligned_free(attention_heads);

    // Layer normalization?

    count2 = get_counter();
    printf("FINISHing CPU Attention Heads Processing...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);
    return attention_heads_out;
}

// Feed forward neural network after attention heads
static int *CPU_EdgeBert_feed_forward(
    int *attention_head_out,
    int input_m,
    int input_n,
    int hidden_size_ffn
) {
    printf("STARTing CPU Feed Forward Net Computation...\n");

    // Profile
    uint64_t count1;
    uint64_t count2;
    count1 = get_counter();

    // Initialize input, weights, and outputs
    int *we_mat1;
    int *we_mat2;
    int *output1;
    int *output2;

    we_mat1 = aligned_malloc(input_n * hidden_size_ffn * sizeof(int));
    we_mat2 = aligned_malloc(hidden_size_ffn * input_n * sizeof(int));
    output1 = aligned_malloc(input_m * hidden_size_ffn * sizeof(int));
    output2 = aligned_malloc(input_m * input_n * sizeof(int));

    // Fill with dummy data
    // CPU_EdgeBert_init_buf_ffn(we_mat1, we_mat2);
    for (int i = 0; i < input_n * hidden_size_ffn; i++) {
        we_mat1[i] = 24;
        we_mat2[i] = -5;
    }

    // Matrix configuration
    int N0;
    int N1;
    int M_mat;
    N0 = input_m; M_mat = input_n; N1 = hidden_size_ffn;

    // First multiplication with attention output
    CPU_multiply(attention_head_out, we_mat1, N0, M_mat, N1, output1);

    // Activation function?

    // Second multiplication
    N0 = input_m; M_mat = hidden_size_ffn; N1 = input_n;
    CPU_multiply(output1, we_mat2, N0, M_mat, N1, output2);

    // Layer normalization?

    count2 = get_counter();
    printf("FINISHing CPU Feed Forward Net Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);

    return output2;
}

// Profile CPU performance
static void CPU_transformer(
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels,
    int hidden_size_ffn
) {
    printf("Transformer Matmul Performance Profiling on Ariane RISC-V CPU BEGIN...\n");
    // Attention heads
    int *attention_heads = CPU_EdgeBert_attention_heads(
        num_heads,
        input_m,
        input_n,
        hidden_size,
        num_labels
    );

    // Processing
    printf("\n");
    int *attention_head_out = CPU_EdgeBert_processing(
        attention_heads,
        num_heads,
        input_m,
        input_n,
        hidden_size
    );

    // Feed Forward Neural Net
    printf("\n");
    int *out = CPU_EdgeBert_feed_forward(
        attention_head_out,
        input_m,
        input_n,
        hidden_size_ffn
    );

    aligned_free(out);

    printf("Transformer Matmul Performance Profiling on Ariane RISC-V CPU DONE...\n");
    printf("Thank you!\n");
}


// Accelerator functions
// Data functions
// Output validation
static int validate_buf(token_t *out, native_t *gold, int out_len) {
    int j;
    native_t val;
    unsigned errors = 0;

    // Iterate over outputs
    for (j = 0; j < out_len; j++) {
        val = out[j];
        // Check for mismatch
        if (gold[j] != val) {
            errors++;
            if (errors <= MAX_PRINTED_ERRORS) {
                printf("%d : %d : %d\n", j, (int) val, (int) gold[j]);
            }
        }
    }
    return errors;
}

// Input initialization
// static void EdgeBert_init_buf_attention(
//     token_t *input_ids,
//     token_t *mask_input,
//     token_t *we_query,
//     token_t *mask_query,
//     token_t *we_key,
//     token_t *mask_key,
//     token_t *we_val,
//     token_t *mask_val,
//     token_t *aux_mat
// ) {
//     #include "data/attention/input_ids.h"     // 128 * 768
//     #include "data/attention/mask_input.h"
//     #include "data/attention/we_query.h"      // 768 * 64
//     #include "data/attention/mask_query.h"
//     #include "data/attention/we_key.h"        // 768 * 64
//     #include "data/attention/mask_key.h"
//     #include "data/attention/we_val.h"        // 768 * 64
//     #include "data/attention/mask_val.h"
//     #include "data/attention/aux_mat.h"
// }

// static void EdgeBert_init_buf_pooler(token_t *we_mat1, token_t *mask_mat1) {
//     #include "data/pooler/we_mat1.h"          // 64 * 64
//     #include "data/pooler/mask_mat1.h"
// }

// static void EdgeBert_init_buf_highway(token_t *we_mat1, token_t *mask_mat1) {
//     #include "data/highway/we_mat1.h"         // 64 * 2
//     #include "data/highway/mask_mat1.h"
// }

// static void EdgeBert_init_buf_processing(token_t *we_mat1, token_t *mask_mat1) {
//     #include "data/processing/we_mat1.h"      // 768 * 768
//     #include "data/processing/mask_mat1.h"
// }

// static void EdgeBert_init_buf_ffn(
//     token_t *we_mat1,
//     token_t *mask_mat1,
//     token_t *we_mat2,
//     token_t *mask_mat2
// ) {
//     #include "data/ffn/we_mat1.h"             // 768 * 3072
//     #include "data/ffn/mask_mat1.h"
//     #include "data/ffn/we_mat2.h"             // 3072 * 768
//     #include "data/ffn/mask_mat2.h"
// }


// Helper functions
// Wait for interrupt
static int wait(struct esp_device *plic_dev, int num_interrupts) {
    printf("......waiting for interrupt #%d\n", num_interrupts + 1);
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    printf("......receiving for interrupt #%d\n", num_interrupts + 1);
    return num_interrupts + 1;
}

// Master mask read (load from outside and store in accelerator MASK scratchpad)
static int master_mask_read(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    unsigned mask_rd_base,
    unsigned decoder,
    unsigned M,
    int num_interrupts
) {
    unsigned data = 0;

    // Not writing to memory
    iowrite32(dev, 0x4C, data);

    // Set mask read address
    data = mask_rd_base;
    iowrite32(dev, 0x28, data);

    // Select decoder
    data = decoder;
    iowrite32(dev, 0x08, data);

    // Set number of words
    data = M - 1;
    iowrite32(dev, 0x40, data);

    // Start master mask read
    data = 0x01;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    return wait(plic_dev, num_interrupts);
}

// Master input read (load from outside and store in accelerator INPUT scratchpad)
static int master_input_read(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    unsigned input_rd_base,
    unsigned decoder,
    unsigned M,
    int num_interrupts
) {
    unsigned data = 0;

    // Not writing to memory
    iowrite32(dev, 0x4C, data);

    // Set base read address
    data = input_rd_base;
    iowrite32(dev, 0x30, data);

    // Set decoder
    data = decoder;
    iowrite32(dev, 0x08, data);

    // Set number of words
    data = M - 1;
    iowrite32(dev, 0x40, data);

    // Start master input read
    data = 0x03;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    return wait(plic_dev, num_interrupts);
}

// Master aux read (load from outside and store in accelerator AUX scratchpad)
static int master_aux_read(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    unsigned aux_rd_base,
    unsigned M,
    int num_interrupts
) {
    unsigned data = 0;

    // Not writing to memory
    iowrite32(dev, 0x4C, data);

    // Set aux read base
    data = aux_rd_base;
    iowrite32(dev, 0x38, data);

    // Set number of words
    data = M - 1;
    iowrite32(dev, 0x44, data);

    // Start master aux read
    data = 0x05;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    return wait(plic_dev, num_interrupts);
}

// Master mask write (load from accelerator MASK scratchpad and store outside)
static int master_mask_write(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    unsigned M,
    int num_interrupts
) {
    unsigned data = 0;

    // Set use_axi to 1
    data = 0x1;
    iowrite32(dev, 0x4C, data);

    // Set number of words
    data = M - 1;
    iowrite32(dev, 0x40, data);

    // Start input master write
    data = 0x02;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    return wait(plic_dev, num_interrupts);
}

// Master input write (load from accelerator DATA scratchpad and store outside)
static int master_input_write(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    unsigned M,
    int num_interrupts
) {
    unsigned data = 0;

    // Set use_axi to 1
    data = 0x1;
    iowrite32(dev, 0x4C, data);

    // Set number of words
    data = M - 1;
    iowrite32(dev, 0x40, data);

    // Start input master write
    data = 0x04;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    return wait(plic_dev, num_interrupts);
}

// Master aux write (load from accelerator AUX scratchpad and store outside)
static int master_aux_write(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    unsigned M,
    int num_interrupts
) {
    unsigned data = 0;

    // Set use_axi to 1
    data = 0x1;
    iowrite32(dev, 0x4C, data);

    // Set number of words
    data = M - 1;
    iowrite32(dev, 0x40, data);

    // Start aux master write
    data = 0x06;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    return wait(plic_dev, num_interrupts);
}

static int load_matrix(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat,
    int decoder,
    int num_interrupts
) {
    // Calculate base addresses
    unsigned mask_rd_base = (uintptr_t) mem;
    unsigned input_rd_base = ((uintptr_t) mem) + 2 * mask_buffer_size;
    unsigned mask_offset = 0;
    unsigned input_offset = 0;
    if (decoder == 1) {
        mask_offset += mask_buffer_size;
        input_offset + input_buffer_size;
    }

    // Load in matrices from memory
    memcpy(mem + mask_offset, mat -> mask, (N0 * M_mat / bits_in_bytes));
    memcpy(mem + 2 * mask_buffer_size + input_offset, mat -> values, N0 * M_mat);

    // Load in matrices to accelerator
    // Master mask read (load from outside and store in accelerator MASK scratchpad) for decoder 0
    int M = N0 * M_mat / vector_size;
    num_interrupts = master_mask_read(dev, plic_dev, mask_rd_base + mask_offset, decoder, M, num_interrupts);
    // Load in data to decoder 0
    num_interrupts = master_input_read(dev, plic_dev, input_rd_base + input_offset, decoder, M, num_interrupts);
    return num_interrupts;
}

static int load_matrices(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int mat1_N0,
    int mat1_M_mat,
    int mat2_N0,
    int mat2_M_mat,
    struct mat *mat1,
    struct mat *mat2,
    int num_interrupts
) {
    // Load in first matrix
    num_interrupts = load_matrix(
        dev,
        plic_dev,
        mem,
        mat1_N0,
        mat1_M_mat,
        mat1,
        0,
        num_interrupts
    );

    // Load in second matrix
    num_interrupts = load_matrix(
        dev,
        plic_dev,
        mem,
        mat2_N0,
        mat2_M_mat,
        mat2,
        1,
        num_interrupts
    );
    return num_interrupts;
}

static int write_matrix(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    int N0,
    int N1,
    int num_interrupts
) {
    int M = N0 * N1 / vector_size;
    num_interrupts = master_input_write(dev, plic_dev, M, num_interrupts);
    num_interrupts = master_mask_write(dev, plic_dev, M, num_interrupts);
    return num_interrupts;
}

// Component functions
// Set base addresses for read and write
static int EdgeBert_init(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem
) {
    printf("...STARTing base addr setting for EdgeBert...\n");

    // TODO: Our reset hack
    iowrite32(dev, 0x00, 1);

    unsigned data = 0;
    int num_interrupts = 0;

    // Calculate base addresses
    // Base address
    unsigned mask_rd1_base = ((uintptr_t) mem);
    unsigned mask_rd2_base = ((uintptr_t) mem) + mask_buffer_size;
    unsigned input_rd1_base = ((uintptr_t) mem) + 2 * mask_buffer_size;
    unsigned input_rd2_base = ((uintptr_t) mem) + 2 * mask_buffer_size + input_buffer_size;
    unsigned aux_rd_base = ((uintptr_t) mem) + 2 * mask_buffer_size + 2 * input_buffer_size;
    unsigned input_wr_base = ((uintptr_t) mem) + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size;
    unsigned mask_wr_base = ((uintptr_t) mem) + 2 * mask_buffer_size + 3 * input_buffer_size + aux_buffer_size;
    unsigned aux_wr_base = ((uintptr_t) mem) + 3 * mask_buffer_size + 3 * input_buffer_size + aux_buffer_size;

    // Set aux read address
    data = aux_rd_base;
    iowrite32(dev, 0x38, data);
    // Set mask write address
    data = mask_wr_base;
    iowrite32(dev, 0x2C, data);
    // Set input write address
    data = input_wr_base;
    iowrite32(dev, 0x34, data);
    // Set aux write address
    data = aux_wr_base;
    iowrite32(dev, 0x3C, data);

    printf("...FINISHing base addr setting for EdgeBert...\n");
}

// Matrix multiplication
static void EdgeBert_mat_mul(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int N1,
    int M_mat,
    struct mat *mat1,
    struct mat *mat2,
    int is_relu,
    int is_bias,
    int weight_bias,
    int write
) {
    printf("...STARTing matmul in EdgeBert...\n");
    int num_interrupts = 0;
    unsigned data = 0;

    // Load in both matrices
    num_interrupts = load_matrices(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        M_mat,
        N1,
        mat1,
        mat2,
        num_interrupts
    );

    // Set basic configs
    // Not using SFU
    data = 0;
    iowrite32(dev, 0x50, data);
    // Use 8-bit MAC
    data = 0;
    iowrite32(dev, 0x60, data);
    data = 0;
    // Set mode to N0
    data = 0x81;
    iowrite32(dev, 0x58, data);

    // Start matrix size configurations
    data += N0;
    data += N1 << 10;
    data += M_mat << 20;
    iowrite32(dev, 0x10, data);

    // Set post-processing configuration
    int accum_right_shift = accum_right_shift_base - mat1 -> bias - mat2 -> bias;
    data = 0;
    data += is_relu;
    data += is_bias << 4;
    data += weight_bias << 8;
    data += adf_accum_bias << 16;
    data += accum_right_shift << 20;
    iowrite32(dev, 0x0C, data);

    // Set base input of first and second matrix
    data = 0;
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);
    // Store base output of computations
    int base_output = input_buffer_size - N0 * N1;
    data = 0;
    data += base_output;
    iowrite32(dev, 0x48, data);

    // Do matrix multiplication
    data = 0x07;
    iowrite32(dev, 0x04, data);
    num_interrupts = wait(plic_dev, num_interrupts);

    // Write data outside
    if (write == 0) {
        num_interrupts = write_matrix(
            dev,
            plic_dev,
            N0,
            N1,
            num_interrupts
        );
    }
    printf("...FINISHing Matmul in EdgeBert...\n");
}

static struct mat *general_mat_mul(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int N1,
    int M_mat,
    struct mat *mat1,
    struct mat *mat2,
    int is_relu,
    int is_bias,
    int weight_bias
) {
    unsigned N0_tile;
    unsigned N1_tile;

    // Try to get as many columns
    N1_tile = mask_buffer_size * bits_in_bytes / M_mat;
    N1_tile = (N1_tile / 16) * 16;
    N1_tile = min(N1_tile, N1);

    N0_tile = input_buffer_size / (M_mat + N1_tile);
    N0_tile = (N0_tile / 16) * 16;
    N0_tile = min(N0_tile, N0);

    // Allocate memory for matrices
    struct mat *left = aligned_malloc(sizeof(struct mat));
    struct mat *right = aligned_malloc(sizeof(struct mat));
    struct mat *output = aligned_malloc(sizeof(struct mat));

    token_t *val_left = aligned_malloc(N0_tile * M_mat);
    token_t *val_right = aligned_malloc(M_mat * N1_tile);
    token_t *val_output = aligned_malloc(N0 * N1);

    token_t *mask_left = aligned_malloc(N0_tile * M_mat / bits_in_bytes);
    token_t *mask_right = aligned_malloc(M_mat * N1_tile / bits_in_bytes);
    token_t *mask_output = aligned_malloc(N0 * N1 / bits_in_bytes);

    // Tranpose for easier access
    CPU_transpose(mat2 -> values, M_mat, N1);

    int row = 0, col = 0;
    while (row < N0) {
        // Get left matrix
        unsigned N0_mat = min(N0_tile, N0 - row);
        memcpy(left -> values, mat1 -> values + M_mat * row, N0_mat * M_mat * sizeof(token_t));
        memcpy(left -> mask, mat1 -> mask + (M_mat * row / bits_in_bytes), N0_mat * M_mat / bits_in_bytes);

        while (col < N1) {
            // Get right matrix
            unsigned N1_mat = min(N1_tile, N1 - col);
            memcpy(right -> values, mat2 -> values + M_mat * col, N1_mat * M_mat * sizeof(token_t));
            memcpy(right -> mask, mat2 -> mask + (M_mat * col / bits_in_bytes), N1_mat * M_mat / bits_in_bytes);
            CPU_transpose(right -> values, N1_mat, M_mat);

            // Multiply
            EdgeBert_mat_mul(
                dev,
                plic_dev,
                mem,
                N0_mat,
                N1_mat,
                M_mat,
                left,
                right,
                is_relu,
                is_bias,
                weight_bias,
                0
            );

            // Copy over data into output
            for (int l = 0; l < N0_mat; l++) {
                for (int k = 0; k < N1_mat; k++) {
                    output -> values[(row + l) * N1 + col + k] = mem[2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * N0_mat + k];
                }
            }

            // Copy over mask into output
            for (int l = 0; l < N0_mat * N1_mat / bits_in_bytes; l++) {
                output -> mask[(row * N1 + col) / bits_in_bytes + l] = mem[2 * mask_buffer_size + 3 * input_buffer_size + aux_buffer_size + l];
            }

            col += N1_mat;
        }
        row += N0_mat;
    }

    aligned_free(val_left);
    aligned_free(mask_left);
    aligned_free(left);
    aligned_free(val_right);
    aligned_free(mask_right);
    aligned_free(right);
    return output;
}

// Softmax for attention head
static void EdgeBert_atten_softmax(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *input,
    token_t *span_mask,
    int base_input
) {
    printf("STARTing attention softmax in EdgeBert...\n");
    unsigned data = 0;
    int num_interrupts = 0;

    // Calculate base addresses
    if (base_input == 0) {
        load_matrix(
            dev,
            plic_dev,
            mem,
            N0,
            M_mat,
            input,
            0,
            num_interrupts
        );
    }

    // Master aux read for mask
    unsigned aux_rd_base = ((uintptr_t) mem) + 2 * mask_buffer_size + 2 * input_buffer_size;
    memcpy(mem + 2 * mask_buffer_size + 2 * input_buffer_size, span_mask, N0 * M_mat / bits_in_bytes);
    int M = N0 * M_mat / vector_size;
    num_interrupts = master_aux_read(dev, plic_dev, aux_rd_base, M, num_interrupts);

    // Choose decoder 0
    data = 0x0;
    iowrite32(dev, 0x08, data);
    // Set SFU to true
    data = 0x1;
    iowrite32(dev, 0x50, data);
    // Set softmax for decoder 0
    data = 0x8;
    iowrite32(dev, 0x58, data);
    // Set mode to softmax
    data = 0x02;
    iowrite32(dev, 0x54, data);

    // Configure matrix size
    data = 0x0;
    data += N0;
    data += M_mat << 20;
    iowrite32(dev, 0x10, data);

    // Configure parameters
    int base_attn_span = 0;
    int adpbias_attn_span = 0;
    data = 0;
    data += base_attn_span;
    data += adpbias_attn_span << 23;
    iowrite32(dev, 0x1C, data);

    int num_vector = M_mat / 16;
    int num_timestep = N0;
    int adpbias_act1 = input -> bias;
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    iowrite32(dev, 0x20, data);

    // Set base of matrix
    data = 0;
    int base_output = input_buffer_size - N0 * M_mat;
    if (base_input != 0) {
        data = base_input;
        base_output = 0;
    }
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);
    // Store base output of computations
    data = 0;
    data += base_output;
    iowrite32(dev, 0x48, data);

    // Start softmax
    data = 0x9;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Master input write
    num_interrupts = write_matrix(
        dev,
        plic_dev,
        N0,
        M_mat,
        num_interrupts
    );
    printf("FINISHing attention softmax in EdgeBert...\n");
}

static void EdgeBert_element_add(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    struct mat *mat2,
    int write
) {
    printf("...STARTing Element Add in EdgeBert...\n");
    int num_interrupts = 0;
    num_interrupts = load_matrices(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        N0,
        M_mat,
        mat1,
        mat2,
        num_interrupts
    );

    // Perform element-wise addition on mat1 + mat2
    unsigned data = 0;
    // Use SFU
    data = 0x1;
    iowrite32(dev, 0x50, data);
    // Set reset_mode to 0b100000'000000
    data = 0x800;
    iowrite32(dev, 0x58, data);
    // Set mode_config to ElemAdd
    data = 0x04;
    iowrite32(dev, 0x54, data);

    // Set activation config
    int num_vector = M_mat / 16;
    int num_timestep = N0;
    int adpbias_act1 = mat1 -> bias;
    int adpbias_act2 = mat2 -> bias;
    int adpbias_act3 = adf_accum_bias;
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    iowrite32(dev, 0x20, data);

    // Do the addition
    // Set matrix config
    data = 0x0;
    data += N0;
    data += M_mat << 20;
    iowrite32(dev, 0x10, data);

    // Set base input
    data = 0;
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);

    // Start element-wise addition
    data = 0xA;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Write data outside
    if (write == 0) {
        num_interrupts = write_matrix(
            dev,
            plic_dev,
            N0,
            M_mat,
            num_interrupts
        );
    }
    printf("...FINISHing Element Add in EdgeBert...\n");
}

// Apply layer norm
static void EdgeBert_layer_norm(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    int base_input
) {
    printf("...STARTing Element Add in EdgeBert...\n");
    unsigned data = 0;
    int num_interrupts = 0;

    if (base_input == 0) {
        load_matrix(
            dev,
            plic_dev,
            mem,
            N0,
            M_mat,
            mat1,
            0,
            num_interrupts
        );
    }

    // Set to decoder 0
    data = 0x0;
    iowrite32(dev, 0x08, data);
    // Enable SFU
    data = 0x1;
    iowrite32(dev, 0x50, data);
    // Set reset_mode to layer norm for decoder 0
    data = 0x04;
    iowrite32(dev, 0x58, data);
    // Set mode to layer norm
    data = 0x1;
    iowrite32(dev, 0x54, data);

    // Set layer norm configs
    int base_gamma = 1;
    int adpbias_gamma = 0;
    int base_beta = 0;
    int adpbias_beta = 0;
    data = 0;
    data += base_gamma << 7;
    data += base_beta << 15;
    data += adpbias_gamma << 26;
    data += adpbias_beta << 29;
    iowrite32(dev, 0x1C, data);

    int num_vector = M_mat / 16;
    int num_timestep = N0;
    int adpbias_act1 = mat1 -> bias;
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    iowrite32(dev, 0x20, data);

    // Set base input
    data = 0;
    int base_output = input_buffer_size - N0 * M_mat;
    if (base_input != 0) {
        data = base_input;
        base_output = 0;
    }
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);
    // Store base output of computations
    data = 0;
    data += base_output;
    iowrite32(dev, 0x48, data);

    // Start layer norm
    data = 0x8;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Write output to outside
    num_interrupts = write_matrix(
        dev,
        plic_dev,
        N0,
        M_mat,
        num_interrupts
    );

    printf("...FINISHing Layer Norm in EdgeBert...\n");
}

static void EdgeBert_entropy(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    int base_input
) {
    return;
}

// // QUESTION: Review architecture?
// static struct mat *EdgeBert_pooler(
//     struct esp_device *dev,
//     struct esp_device *plic_dev,
//     token_t *mem,
//     struct mat *attention_heads,
//     int input_m,
//     int hidden_size
// ) {
//     // Weight matrix (hidden_size * hidden_size)
//     struct mat *we_mat1 = aligned_malloc(sizeof(struct mat));
//     token_t *val_mat1 = aligned_malloc(hidden_size * hidden_size * sizeof(token_t));
//     token_t *mask_mat1 = aligned_malloc(hidden_size * hidden_size / bits_in_bytes);
//     int bias_mat1 = 0;

//     // Fill with data
//     // EdgeBert_init_buf_pooler(val_mat1, mask_mat1, bias_mat1);
//     // TODO: Add memset
//     *we_mat1 = (struct mat) {val_mat1, mask_mat1, bias_mat1};

//     // Matrix multiplication configurations
//     int N0;
//     int N1;
//     int M_mat;
//     int is_relu;
//     int is_bias;
//     int weight_bias;
//     int softmax;

//     N0 = input_m; M_mat = hidden_size; N1 = hidden_size;
//     is_relu = 0, is_bias = 0, weight_bias = 0, softmax = 0;

//     // Query multiplication
//     general_mat_mul(
//         dev,
//         plic_dev,
//         mem,
//         N0,
//         N1,
//         M_mat,
//         attention_heads,
//         we_mat1,
//         is_relu,
//         is_bias,
//         weight_bias,
//         softmax
//     );

//     // Activation?

//     // TODO: Fix architecture
//     // Get hidden states for first token
//     struct mat *out = aligned_malloc(sizeof(struct mat));
//     token_t *val_out = aligned_malloc(hidden_size);
//     token_t *mask_out = aligned_malloc(hidden_size / 8);

//     // Copy over data
//     memcpy(out, mem + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, hidden_size);
//     out = (struct mat) {val_out, mask_out, accum_right_shift};

//     // Free allocated space
//     aligned_free(val_mat1);
//     aligned_free(mask_mat1);
//     aligned_free(we_mat1);
//     return out;
// }

// static token_t *EdgeBert_highway_exit(
//     struct esp_device *dev,
//     struct esp_device *plic_dev,
//     token_t *mem,
//     struct mat *attention_heads,
//     int input_m,
//     int hidden_size,
//     int num_labels
// ) {
//     // Highway exit
//     // Index in to layer outputs
//     token_t *pooler_output = EdgeBert_pooler(
//         dev,
//         plic_dev,
//         mem,
//         attention_heads,
//         input_m,
//         hidden_size
//     );

//     // Intialize linear layer and outputs
//     struct mat *we_mat1 = aligned_malloc(sizeof(struct mat));
//     token_t *val_mat1 = aligned_malloc(num_labels * hidden_size * sizeof(token_t));
//     token_t *mask_mat1 = aligned_malloc(num_labels * hidden_size / bits_in_bytes);

//     struct mat *outpt = aligned_malloc(sizeof(struct mat));
//     token_t *val_output;
//     token_t *mask_ouptut;

//     // Fill with data
//     EdgeBert_init_buf_highway(we_mat1);

//     // Initialize config
//     int N0;
//     int N1;
//     int M_mat;
//     unsigned is_relu;
//     int softmax;

//     N0 = num_labels;
//     M_mat = hidden_size;
//     N1 = 1;
//     is_relu = 0;
//     softmax = 0;

//     // Perform matrix multiplication
//     general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat1, mask_mat2, we_mat1, pooler_output, softmax);
//     memcpy(result_mat_1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

//     aligned_free(we_mat1);
//     aligned_free(mask_mat1);
//     aligned_free(mask_mat2);

//     return output;
// }

// // Attention head
// static token_t *EdgeBert_attention(
//     struct esp_device *dev,
//     struct esp_device *plic_dev,
//     token_t *mem,
//     int input_m,
//     int input_n,
//     int hidden_size
// ) {
//     printf("STARTing Attention Head in EdgeBert...\n");

//     int num_interrupts;

//     // Initialize inputs and weights
//     token_t *input_ids;
//     token_t *mask_input;
//     token_t *we_query;
//     token_t *mask_query;
//     token_t *we_key;
//     token_t *mask_key;
//     token_t *we_val;
//     token_t *mask_val;
//     token_t *aux_mat;

//     // Initialize IDs
//     input_ids = aligned_malloc(input_m * input_n);
//     mask_input = aligned_malloc(ceil_div(input_m * input_n, 8));
//     we_query = aligned_malloc(input_n * hidden_size);
//     mask_query = aligned_malloc(ceil_div(input_n * hidden_size, 8));
//     we_key = aligned_malloc(input_n * hidden_size);
//     mask_key = aligned_malloc(ceil_div(input_n * hidden_size, 8));
//     we_val = aligned_malloc(input_n * hidden_size);
//     mask_val = aligned_malloc(ceil_div(input_n * hidden_size, 8));

//     // QUESTION: Size?
//     aux_mat = aligned_malloc(4096);

//     // Initialize weights
//     // EdgeBert_init_buf_attention(
//     //     input_ids,
//     //     mask_input,
//     //     we_query,
//     //     mask_query,
//     //     we_key,
//     //     mask_key,
//     //     we_val,
//     //     mask_val,
//     //     aux_mat
//     // );
//     memset(input_ids, 11, input_m * input_n * sizeof(token_t));
//     memset(mask_input, 255, ceil_div(input_m * input_n, 8) * sizeof(token_t));
//     memset(we_query, 35, input_n * hidden_size * sizeof(token_t));
//     memset(mask_query, 255, ceil_div(input_n * hidden_size, 8)* sizeof(token_t));
//     memset(we_key, -1, input_n * hidden_size * sizeof(token_t));
//     memset(mask_key, 255, ceil_div(input_n * hidden_size, 8) * sizeof(token_t));
//     memset(we_val, -12, input_n * hidden_size * sizeof(token_t));
//     memset(mask_val, 255, ceil_div(input_n * hidden_size, 8) * sizeof(token_t));
//     memset(aux_mat, 3, 4096 * sizeof(token_t));

//     // Initialize matrix multiplication
//     unsigned N0;
//     unsigned N1;
//     unsigned M_mat;
//     unsigned is_relu;
//     int softmax;

//     N0 = input_m;
//     M_mat = input_n;
//     N1 = hidden_size;
//     is_relu = 0;
//     softmax = 0;

//     token_t *query_mat;
//     token_t *mask_query_res;
//     token_t *key_mat;
//     token_t *mask_key_res;
//     token_t *val_mat;
//     token_t *mask_val_res;

//     // Initialize output matrices
//     query_mat = aligned_malloc(N0 * N1);
//     mask_query_res = aligned_malloc();
//     key_mat = aligned_malloc(N0 * N1);
//     mask_key_res = aligned_malloc();
//     val_mat = aligned_malloc(N0 * N1);
//     mask_val_res = aligned_malloc();

//     // Mutliply IDs by query matrix
//     general_mat_mul(
//         dev,
//         plic_dev,
//         N0,
//         N1,
//         M_mat,
//         is_relu,
//         mem,
//         mask_input,
//         mask_query,
//         input_ids,
//         we_query,
//         softmax
//     );
//     // Save result
//     memcpy(query_mat, mem + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1);

//     // Mutliply IDs by key matrix
//     general_mat_mul(
//         dev,
//         plic_dev,
//         N0,
//         N1,
//         M_mat,
//         is_relu,
//         mem,
//         mask_input,
//         mask_key,
//         input_ids,
//         we_key,
//         softmax
//     );
//     // Save result
//     memcpy(key_mat, mem + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1);

//     // Mutliply IDs by value matrix
//     general_mat_mul(
//         dev,
//         plic_dev,
//         N0,
//         N1,
//         M_mat,
//         is_relu,
//         mem,
//         mask_input,
//         mask_val,
//         input_ids,
//         we_val,
//         softmax
//     );
//     // Save result
//     memcpy(val_mat, mem + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1);

//     // Transpose output of key multiplication
//     CPU_transpose(key_mat, N0, N1);

//     // Matrix config
//     N0 = input_m;
//     M_mat = input_n;
//     N1 = input_m;
//     // Set softmax parameter to true
//     softmax = 1;

//     token_t *query_key_mat;
//     query_key_mat = aligned_malloc(N0 * N1);

//     // Multiply query and key outpput
//     general_mat_mul(
//         dev,
//         plic_dev,
//         N0,
//         N1,
//         M_mat,
//         is_relu,
//         mem,
//         mask_mat,
//         query_mat,
//         key_mat,
//         softmax
//     );

//     // Free memory
//     aligned_free(query_mat);
//     aligned_free(key_mat);

//     // Softmax and attention span configuration
//     N0 = input_m;
//     M_mat = input_m;
//     N1 = input_m;

//     // Apply softmax
//     EdgeBert_atten_softmax(dev, plic_dev, N0, N1, M_mat);
//     memcpy(query_key_mat, mem + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

//     // Multiply query and key with value matrix
//     N0 = input_m;
//     M_mat = input_m;
//     N1 = hidden_size;
//     softmax = 0;

//     general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, query_key_mat, val_mat, softmax);
//     memcpy(val_mat, mem + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

//     // Free memory
//     aligned_free(input_ids);
//     aligned_free(we_query);
//     aligned_free(we_key);
//     aligned_free(we_val);
//     aligned_free(mask_mat);
//     aligned_free(aux_mat);
//     aligned_free(query_key_mat);

//     printf("FINISHing Attention Head in EdgeBert...\n");
//     return val_mat;
// }

// static token_t *EdgeBert_attention_heads(
//     struct esp_device *dev,
//     struct esp_device *plic_dev,
//     token_t *mem,
//     int num_heads,
//     int input_m,
//     int input_n,
//     int hidden_size,
//     int num_labels
// ) {
//     printf("STARTing EdgeBERT %d Attention Heads Computation...\n", num_heads);

//     // Profiling
//     uint64_t total_exe_cycle = 0;
//     uint64_t count1;
//     uint64_t count2;
//     uint64_t exe_cycle;

//     token_t* attention_heads;
//     attention_heads = aligned_malloc(input_m * hidden_size * num_heads);

//     for (int i = 0; i < 12; i++) {
//         // Run attention head
//         count1 = get_counter();
//         EdgeBert_attention(
//             dev,
//             plic_dev,
//             mem,
//             input_m,
//             input_n,
//             hidden_size
//         );
//         count2 = get_counter();
//         exe_cycle = count2 - count1;
//         printf("...Attention Head %d takes %"PRIu64" clock cycles...\n", i, exe_cycle);

//         for (int l = 0; l < input_m; l++) {
//             for (int k = 0; k < hidden_size; k++) {
//                 attention_heads[l * hidden_size * num_heads + i * hidden_size + k] = mem[2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * hidden_size + k];
//             }
//         }

//         // Keep track of number of cycles
//         total_exe_cycle = total_exe_cycle + exe_cycle;
//     }

//     printf("FINISHing EdgeBERT 12 Attention Heads Computation...\n");
//     printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
//     return attention_heads;
// }

// static void EdgeBert_processing(
//     struct esp_device *dev,
//     struct esp_device *plic_dev,
//     token_t *mem,
//     token_t *attention_heads,
//     int num_heads,
//     int input_m,
//     int input_n,
//     int num_labels
// ) {
//     printf("STARTing 12 Attention Heads Processing...\n");

//     // Profiling
//     uint64_t total_exe_cycle = 0;
//     uint64_t count1;
//     uint64_t count2;
//     uint64_t exe_cycle;

//     count1 = get_counter();
//     // Multiply concatenated output (128 x 768)
//     token_t* we_mat1;
//     we_mat1 = aligned_malloc(768 * 768);
//     // Fill with dummy data
//     EdgeBert_init_buf_processing(we_mat1);

//     // QUESTION: Should mask also be included in the data?
//     token_t *mask_mat;
//     mask_mat = aligned_malloc(8192);
//     // Fill with dummy data
//     memset(mask_mat, 255, 8192 * sizeof(token_t));

//     // Use accelerator and split input into two
//     token_t *input_1;
//     token_t *input_2;

//     // Matrix multiplication configurations
//     unsigned N0;
//     unsigned N1;
//     unsigned M_mat;
//     unsigned is_relu;

//     N0 = 64;
//     M_mat = 768;
//     N1 = 64;
//     is_relu = 0;

//     input_1 = aligned_malloc(N0 * M_mat);
//     input_2 = aligned_malloc(M_mat * N1);

//     EdgeBert_init(&dev, &plic_dev, mem);
//     int count = 0;
//     for (int i = 0; i < 2; i++) {
//         memcpy(input_1, attention_heads + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));
//         for (int j = 0; j < 12; j++) {
//             memcpy(input_2, We_heads + j * M_mat * N1, M_mat * N1 * sizeof(token_t));

//             if (count == 2) {
//                 EdgeBert_init(&dev, &plic_dev, mem);
//                 count = 0;
//             }

//             EdgeBert_mat_mul (&dev, &plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_1, input_2, 0);
//             for (int l = 0; l < N0; l++) {
//                 for (int k = 0; k < N1; k++) {
//                     attention_head_out[(l + i * N0) * 768 + j * N1 + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + k + l * N0];
//                 }
//             }
//             count++;
//         }
//     }

//     aligned_free(input_1);
//     aligned_free(input_2);

//     N0 = 64;
//     M_mat = 768;
//     N1 = 64;

//     token_t* attention_out;
//     input_1 = aligned_malloc(N0 * M_mat);
//     input_2  = aligned_malloc(M_mat * N1);

//     // Add on input IDs and layer norm
//     token_t* input_ids;
//     input_ids = aligned_malloc(128 * 768);
//     memset(input_ids, -1, 128 * 768 * sizeof(token_t));
//     CPU_transpose(input_ids, 768, 128);
//     attention_out = aligned_malloc(128 * 768);

//     for (int i = 0; i < 2; i++) {
//         // Split 128 x 768 into two 64 x 768
//         memcpy(input_1, attention_head_out + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));
//         memcpy(input_2, input_ids + i * N1 * M_mat, N1 * M_mat * sizeof(token_t));
//         EdgeBert_init(&dev, &plic_dev, mem);
//         EdgeBert_element_add_layer_norm(&dev, &plic_dev, N0, N1, M_mat, mem, input_1, input_2);
//         memcpy(attention_out + i * N0 * M_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * M_mat * sizeof(token_t));
//     }
//     count2 = get_counter();
//     printf("FINISHing 12 Attention Heads Processing...\n");
//     printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);

//     return attention_out;
// }

// // Feed forward
// static token_t *EdgeBert_feed_forward(
//     struct esp_device *dev,
//     struct esp_device *plic_dev,
//     token_t *mem,
//     token_t *attention_head_out,
//     int input_m,
//     int input_n,
//     int hidden_size_ffn
// ) {
//     printf("\nSTARTing EdgeBERT Feed Forward Net Computation...\n");
//     uint64_t total_exe_cycle = 0;
//     uint64_t count1;
//     uint64_t count2;
//     uint64_t exe_cycle;

//     count1 = get_counter();
//     EdgeBert_init(&dev, &plic_dev, mem);

//     int softmax = 0;
//     token_t *mask_mat;
//     mask_mat = aligned_malloc(8192);
//     memset(mask_mat, 255, 8192 * sizeof(token_t));

//     // Initialize weights
//     token_t* we_mat1; // 768 x 3072
//     token_t* we_mat2; // 3072 x 768

//     we_mat1 = aligned_malloc(768 * 3072);
//     we_mat2 = aligned_malloc(768 * 3072);

//     // init_buf_ffn(we_mat1, we_mat2);
//     // Load dummy data
//     memset(we_mat1, -1, 768 * 3072 * sizeof(token_t));
//     memset(we_mat2, -12, 768 * 3072 * sizeof(token_t));

//     // Multiply attention output by weights ((128 x 768) x (768 x 3072) = (128 x 3072))
//     unsigned N0;
//     unsigned N1;
//     unsigned M_mat;
//     unsigned is_relu;

//     N0 = 64;
//     M_mat = 768;
//     N1 = 64;
//     is_relu = 1;

//     // Split attention output into two
//     token_t *input_1;     // 64 x 768
//     token_t *input_2;     // 768 x 64
//     token_t *relu_output; // 128 x 3072

//     input_1 = aligned_malloc(N0 * M_mat);
//     input_2 = aligned_malloc(M_mat * N1);
//     relu_output = aligned_malloc(128 * 3072);

//     EdgeBert_init(dev, plic_dev, mem);
//     int count = 0;

//     for (int i = 0; i < 2; i++) {
//         // Load in half of attention output
//         memcpy(input_1, processing_out + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));

//         for (int j = 0; j < 48; j++) {
//             // TODO: Need to transpose?
//             memcpy(input_2, we1_mat + j * M_mat * N1, M_mat * N1 * sizeof(token_t));

//             if (count == 2) {
//                 EdgeBert_init(dev, plic_dev, mem);
//                 count = 0;
//             }
//             EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_1, input_2, softmax);
//             // memcpy(relu_output + N0 * N1 * i * j, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

//             // Copy over data into relu_output
//             for (int l = 0; l < N0; l++) {
//                 for (int k = 0; k < N1; k++) {
//                     relu_output[(l + i * N0) * 3072 + j * N1 + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * N0 + k];
//                 }
//             }
//             count++;
//         }
//     }

//     // Multiply relu_out with second weights ((8 x 16) x 3072) x (3072 x (16 x 48))
//     N0 = 16;
//     M_mat = 3072;
//     N1 = 16;
//     is_relu = 0;

//     aligned_free(input_1);
//     aligned_free(input_2);
//     token_t *we2_output;

//     input_1 = aligned_malloc(N0 * M_mat);
//     input_2 = aligned_malloc(M_mat * N1);
//     we2_output = aligned_malloc(128 * 768);

//     // EdgeBert_init(dev, plic_dev, mem);
//     count = 0;

//     for (int i = 0; i < 8; i++) {
//         memcpy(input_1, relu_output + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));

//         for (int j = 0; j < 48; j++) {
//             memcpy(input_2, we2_mat + j * M_mat * N1, M_mat * N1 * sizeof(token_t));

//             if (count == 2) {
//                 // EdgeBert_init(dev, plic_dev, mem);
//                 count = 0;
//             }

//             EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_1, input_2, softmax);
//             // memcpy(we2_output + N0 * N1 * i * j, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

//             for (int l = 0; l < N0; l++) {
//                 for (int k = 0; k < N1; k++) {
//                     we2_output[(l + i * N0) * 768 + j * N1 + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * N0 + k];
//                 }
//             }
//             count++;
//         }
//     }

//     aligned_free(input_1);
//     aligned_free(input_2);

//     N0 = 64;
//     M_mat = 768;
//     N1 = 64;

//     // Add attention output
//     token_t* FFN_output;
//     input_1 = aligned_malloc(N0 * M_mat);
//     input_2  = aligned_malloc(M_mat * N1);
//     CPU_transpose(processing_out, 768, 128);
//     FFN_output = aligned_malloc(128*768);

//     for (int i = 0; i < 2; i++) {
//         // Add parts of attention output
//         memcpy(input_1, we2_output + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));
//         memcpy(input_2, processing_out + i * N1 * M_mat, N1 * M_mat * sizeof(token_t));
//         EdgeBert_init(dev, plic_dev, mem);
//         EdgeBert_ElementAddLayerNorm(dev, plic_dev, N0, N1, M_mat, mem, input_1, input_2);
//         memcpy(FFN_output + i * N0 * M_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * M_mat * sizeof(token_t));
//     }

//     // Free memory
//     aligned_free(input_1);
//     aligned_free(input_2);
//     aligned_free(we2_output);
//     aligned_free(mask_mat);
//     aligned_free(we1_mat);
//     aligned_free(we2_mat);
//     aligned_free(relu_output);

//     count2 = get_counter();
//     printf("FINISHing EdgeBERT Feed Forward Net Computation...\n");
//     printf("###(taking %"PRIu64" clock cycles)###...\n", count2 - count1);
// }

// static void EdgeBert_transformer(
//     struct esp_device *dev,
//     struct esp_device *plic_dev,
//     token_t *mem,
//     int num_heads,
//     int input_m,
//     int input_n,
//     int hidden_size,
//     int num_labels,
//     int hidden_size_ffn
// ) {
//     printf("Transformer Matmul Performance on EdgeBERT BEGIN...\n");
//     EdgeBert_init(dev, plic_dev, mem);
//     // Attention heads
//     printf("\n");
//     token_t *attention_heads = EdgeBert_attention_heads(
//         dev,
//         plic_dev,
//         mem,
//         num_heads,
//         input_m,
//         input_n,
//         hidden_size,
//         num_labels
//     );

//     // Processing
//     printf("\n");
//     token_t *attention_head_out = EdgeBert_processing(
//         dev,
//         plic_dev,
//         mem,
//         attention_heads,
//         num_heads,
//         input_m,
//         input_n,
//         hidden_size
//     );

//     // Feed Forward Neural Net
//     printf("\n");
//     token_t *out = EdgeBert_feed_forward(
//         dev,
//         plic_dev,
//         mem,
//         attention_head_out,
//         input_m,
//         input_n,
//         hidden_size_ffn
//     );

//     aligned_free(out);

//     printf("Transformer Matmul Performance on EdgeBERT FINISH...\n");
//     printf("Thank you!\n");
// }

static void EdgeBert_debugging_matmul(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem
) {
    int num_interrupts = 0;

    // Initialize weights
    token_t *we_mat1;
    token_t *input;
    token_t *mask_mat1;
    token_t *mask_input;

    we_mat1 = aligned_malloc(16 * 16);
    input = aligned_malloc(16 * 16);
    mask_mat1 = aligned_malloc(32);
    mask_input = aligned_malloc(32);

    // Load dummy data
    memset(we_mat1, 0b01001001, 16 * 16);
    memset(input, 0b01001001, 16 * 16);
    memset(mask_mat1, 255, 32);
    memset(mask_input, 255, 32);

    EdgeBert_init(dev, plic_dev, mem);
    int N0 = 16, M_mat = 16, N1 = 16;
    EdgeBert_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        &(struct mat) {we_mat1, mask_mat1, 0},
        &(struct mat) {input, mask_input, 0},
        0,
        0,
        0,
        0
    );

    printf("matmul output\n");
    for (int i = 0; i < mem_size; i++) {
        if (mem[i] != 0 && mem[i] != 2) {
            printf("%d %d\n", i, mem[i]);
        }
    }

    aligned_free(we_mat1);
    aligned_free(input);
    aligned_free(mask_mat1);
    aligned_free(mask_input);
}

static void EdgeBert_debugging_softmax(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem
) {
    int num_interrupts = 0;

    // Initialize weights
    token_t *we_mat1;
    token_t *mask_mat1;
    token_t *mask_span_mask;

    we_mat1 = aligned_malloc(16 * 16);
    mask_mat1 = aligned_malloc(32);
    mask_span_mask = aligned_malloc(32);

    // Load dummy data
    memset(we_mat1, 0b01001001, 16 * 16);
    memset(mask_mat1, 255, 32);
    memset(mask_span_mask, 255, 32);

    EdgeBert_init(dev, plic_dev, mem);
    int N0 = 16, M_mat = 16;
    EdgeBert_atten_softmax(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        &(struct mat) {we_mat1, mask_mat1, 0},
        mask_span_mask,
        0
    );

    printf("softmax output\n");
    for (int i = 0; i < mem_size; i++) {
        if (mem[i] != 0 && mem[i] != 2) {
            printf("%d %d\n", i, mem[i]);
        }
    }

    aligned_free(we_mat1);
    aligned_free(mask_mat1);
}

static void EdgeBert_debugging_element_add(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem
) {
    int num_interrupts = 0;

    // Initialize weights
    token_t *we_mat1;
    token_t *input;
    token_t *mask_mat1;
    token_t *mask_input;

    we_mat1 = aligned_malloc(16 * 16);
    input = aligned_malloc(16 * 16);
    mask_mat1 = aligned_malloc(32);
    mask_input = aligned_malloc(32);

    // Load dummy data
    memset(we_mat1, 0b01001001, 16 * 16);
    memset(input, 0b01001001, 16 * 16);
    memset(mask_mat1, 255, 32);
    memset(mask_input, 255, 32);

    EdgeBert_init(dev, plic_dev, mem);
    int N0 = 16, M_mat = 16;
    EdgeBert_element_add(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        &(struct mat) {we_mat1, mask_mat1, 0},
        &(struct mat) {input, mask_input, 0},
        0
    );

    printf("element add output\n");
    for (int i = 0; i < mem_size; i++) {
        if (mem[i] != 0 && mem[i] != 2) {
            printf("%d %d\n", i, mem[i]);
        }
    }

    aligned_free(we_mat1);
    aligned_free(input);
    aligned_free(mask_mat1);
    aligned_free(mask_input);
}

static void EdgeBert_debugging_layer_norm(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem
) {
    int num_interrupts = 0;

    // Initialize weights
    token_t *we_mat1;
    token_t *mask_mat1;

    we_mat1 = aligned_malloc(16 * 16);
    mask_mat1 = aligned_malloc(32);

    // Load dummy data
    memset(we_mat1, 0b01001001, 16 * 16);
    memset(mask_mat1, 255, 32);

    EdgeBert_init(dev, plic_dev, mem);
    int N0 = 16, M_mat = 16;
    EdgeBert_layer_norm(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        &(struct mat) {we_mat1, mask_mat1, 0},
        0
    );

    printf("layer norm output\n");
    for (int i = 0; i < mem_size; i++) {
        if (mem[i] != 0 && mem[i] != 2) {
            printf("%d %d\n", i, mem[i]);
        }
    }

    aligned_free(we_mat1);
    aligned_free(mask_mat1);
}

// Driver
// Edgebert compuatation
int main(int argc, char * argv[]) {
    printf("STARTing DRIVER\n");
    int i;
    int n;
    int ndev;

    // Initialize device and coherence device
    struct esp_device dev, coh_dev;
    dev.addr = ACC_ADDR;

    // Initialize PLIC
    struct esp_device plic_dev;
    plic_dev.addr = PLIC_ADDR;

    unsigned done;
    token_t *mem;

    unsigned errors1 = 0;
    unsigned errors2 = 0;
    unsigned coherence;
    unsigned data = 0;

    // Accelerator estimation
    // Total mem size
    mem_size = 3 * mask_buffer_size + 3 * input_buffer_size + 2 * aux_buffer_size;
    // Allocation of the accelerator data array (mem)
    mem = aligned_malloc(mem_size);
    memset(mem, 2, mem_size);

    // Flush (customize coherence model here)
    coherence = ACC_COH_RECALL;
    coh_dev.addr = CSR_TILE_ADDR;
    iowrite32(&coh_dev, CSR_REG_OFFSET * 4, coherence);
    if (coherence != ACC_COH_RECALL) {
        esp_flush(coherence, 4);
    }

    printf("  #######   ######      ######       ####    #     #    #####   \n");
    printf("  #         #     #    #      #     #        #     #   #     #  \n");
    printf("  #         #     #   #        #   #         #     #   #        \n");
    printf("  #######   ######    #        #   #         #######    #####   \n");
    printf("  #         #         #        #   #         #     #         #  \n");
    printf("  #         #          #      #     #        #     #   #     #  \n");
    printf("  #######   #           ######       ####    #     #    #####   \n");

    // Run transformer on CPU
    // CPU_transformer(
    //     12,
    //     128,
    //     768,
    //     64,
    //     2,
    //     3072
    // );

    // // Run transformer on accelerator
    // EdgeBert_transformer(
    //     &dev,
    //     &plic_dev,
    //     mem,
    //     12,
    //     128,
    //     768,
    //     64,
    //     2,
    //     3072
    // );

    EdgeBert_debugging_softmax(&dev, &plic_dev, mem);

    printf("FINISHing DRIVER\n");
    aligned_free(mem);
    return 0;
}
