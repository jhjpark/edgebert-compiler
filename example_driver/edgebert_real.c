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

#define PLIC_ADDR 0x6c000000
#define PLIC_IP_OFFSET 0x1000
#define PLIC_INTACK_OFFSET 0x200004
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
const static int mask_scale = 4;

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
    int *new_array = aligned_malloc(m * n);
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
    aligned_free(new_array);
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
    token_t *new_array = aligned_malloc(m * n);

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
    aligned_free(new_array);
}

void CPU_transpose_mask(token_t *mask, int m, int n) {
    token_t *new_mask = aligned_malloc(m * n / bits_in_bytes);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int mask_idx1 = (i * n + j) / 8;
            int offset1 = (i * n + j) % 8;
            int mask_idx2 = (j * m + i) / 8;
            int offset2 = (j * m + i) % 8;

            new_mask[mask_idx2] |= ((mask[mask_idx1] >> offset1) & 1) << offset2;
        }
    }

    // Replace in place
    for (int i = 0; i < m * n / bits_in_bytes; i++) {
        mask[i] = new_mask[i];
    }
    aligned_free(new_mask);
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
    int *we_mat1 = aligned_malloc(hidden_size * hidden_size * sizeof(int));
    int *output = aligned_malloc(input_m * hidden_size * sizeof(int));

    // Fill with data
    // CPU_EdgeBert_init_buf_pooler(we_mat1);
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        we_mat1[i] = 10;
    }

    // Matrix multiplication configurations
    int N0 = input_m;
    int N1 = hidden_size;
    int M_mat = hidden_size;

    // Query multiplication
    CPU_multiply(we_mat1, attention_heads, N0, M_mat, N1, output);

    // Activation?

    // Get hidden states for first token
    int *out = aligned_malloc(hidden_size * sizeof(int));
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
    int *we_mat1 = aligned_malloc(num_labels * hidden_size * sizeof(int));
    int *output1 = aligned_malloc(num_labels * sizeof(int));

    // Fill with data
    // CPU_EdgeBert_init_buf_highway(we_mat1);
    for (int i = 0; i < num_labels * hidden_size; i++) {
        we_mat1[i] = 4;
    }

    // Mutliply pooler output by first matrix
    int N0 = num_labels;
    int N1 = hidden_size;
    int M_mat = 1;
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
    int *input_ids,
    int input_m,
    int input_n,
    int hidden_size
) {
    printf("STARTing Attention Head in CPU...\n");
    // Initialize inputs, weights, and outputs
    int *we_query = aligned_malloc(input_n * hidden_size * sizeof(int));
    int *we_key = aligned_malloc(input_n * hidden_size * sizeof(int));
    int *we_val = aligned_malloc(input_n * hidden_size * sizeof(int));
    int *output1 = aligned_malloc(input_m * hidden_size * sizeof(int));
    int *output2 = aligned_malloc(input_m * hidden_size * sizeof(int));
    int *output3 = aligned_malloc(input_m * hidden_size * sizeof(int));

    // Load in data
    // CPU_EdgeBert_init_buf_attention(input_ids, we_query, we_key, we_val);
    for (int i = 0; i < input_n * hidden_size; i++) {
        we_query[i] = 24;
        we_key[i] = -5;
        we_val[i] = 126;
    }

    // Matrix multiplication configurations
    int N0 = input_m;
    int N1 = input_n;
    int M_mat = hidden_size;

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
    int *output4 = aligned_malloc(input_m * input_m * sizeof(int));
    CPU_multiply(output1, output2, N0, M_mat, N1, output4);

    // Softmax?

    // Attention Span Mask?

    // Multiply value output
    N0 = input_m; M_mat = input_m; N1 = hidden_size;
    int *output5 = aligned_malloc(input_m * hidden_size * sizeof(int));
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
    int *input_ids,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels
) {
    printf("STARTing CPU %d Attention Heads Computation...\n", num_heads);

    // Initialize outputs
    int *attention_heads = aligned_malloc(input_m * input_n * sizeof(int));

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
            input_ids,
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

        aligned_free(attention_output);
    }

    aligned_free(input_ids);
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
    int *we_mat1 = aligned_malloc(hidden_size * num_heads * input_n * sizeof(int));;
    int *attention_heads_out = aligned_malloc(input_m * input_n * sizeof(int));;

    // Fill matrix with data
    // CPU_EdgeBert_init_buf_processing(we_mat1);
    for (int i = 0; i < hidden_size * num_heads * input_n; i++) {
        we_mat1[i] = -1;
    }

    // Multiply output by matrix
    int N0 = input_m;
    int N1 = hidden_size * num_heads;
    int M_mat = input_n;
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
    int *we_mat1 = aligned_malloc(input_n * hidden_size_ffn * sizeof(int));
    int *we_mat2 = aligned_malloc(hidden_size_ffn * input_n * sizeof(int));
    int *output1 = aligned_malloc(input_m * hidden_size_ffn * sizeof(int));
    int *output2 = aligned_malloc(input_m * input_n * sizeof(int));

    // Fill with dummy data
    // CPU_EdgeBert_init_buf_ffn(we_mat1, we_mat2);
    for (int i = 0; i < input_n * hidden_size_ffn; i++) {
        we_mat1[i] = 24;
        we_mat2[i] = -5;
    }

    // Matrix configuration
    int N0 = input_m;
    int N1 = input_n;
    int M_mat = hidden_size_ffn;

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
static int *CPU_transformer(
    int *input,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels,
    int hidden_size_ffn
) {
    printf("STARTing Transformer Layer in CPU...\n");
    // Attention heads
    int *attention_heads = CPU_EdgeBert_attention_heads(
        input,
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

    printf("FINISHing Transformer Layer in CPU...\n");
    return out;
}

static void CPU_transformer_layers(
    int num_layers,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels,
    int hidden_size_ffn
) {
    printf("Transformer Matmul Performance Profiling on Ariane RISC-V CPU BEGIN...\n");
    // Initialize inputs
    int *input = aligned_malloc(input_m * input_n * sizeof(int));
    for (int i = 0; i < input_m * input_n; i++) {
        input[i] = 12;
    }

    for (int i = 0; i < num_layers; i++) {
        input = CPU_transformer(
            input,
            num_heads,
            input_m,
            input_n,
            hidden_size,
            num_labels,
            hidden_size_ffn
        );
    }

    aligned_free(input);

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
    // printf("......waiting for interrupt #%d\n", num_interrupts + 1);
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving for interrupt #%d\n", num_interrupts + 1);
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
    // data = 32;
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
    iowrite32(dev, 0x44, data);
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
    int M = M_mat / vector_size;
    // Master mask read (load from outside and store in accelerator MASK scratchpad) for decoder 0
    num_interrupts = master_mask_read(dev, plic_dev, mask_rd_base + mask_offset, decoder, M, num_interrupts);
    M = M / mask_scale;
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

static struct mat *write_matrix(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int N1,
    struct mat *output,
    int num_interrupts
) {
    if (output == NULL) {
        output = aligned_malloc(sizeof(struct mat));
        token_t *val_output = aligned_malloc(N0 * N1);
        token_t *mask_output = aligned_malloc(N0 * N1 / bits_in_bytes);
        int bias_output = adf_accum_bias;
        *output = (struct mat) {val_output, mask_output, bias_output};
    }

    int M = N1 / vector_size;
    num_interrupts = master_input_write(dev, plic_dev, M, num_interrupts);
    memcpy(output -> values, mem + 2 * mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1);

    num_interrupts = master_mask_write(dev, plic_dev, M, num_interrupts);
    memcpy(output -> mask, mem + 2 * mask_buffer_size + 3 * input_buffer_size + aux_buffer_size, N0 * N1 / bits_in_bytes);

    return output;
}

// Component functions
// Set base addresses for read and write
static int EdgeBert_init(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem
) {
    // TODO: Our reset hack
    unsigned data = 0;
    iowrite32(dev, 0x00, 1);

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
    int write,
    struct mat *output
) {
    EdgeBert_init(dev, plic_dev, mem);
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
        write_matrix(
            dev,
            plic_dev,
            mem,
            N0,
            N1,
            output,
            num_interrupts
        );
    }
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
    struct mat *output
) {
    EdgeBert_init(dev, plic_dev, mem);
    int num_interrupts = 0;
    unsigned data = 0;

    // Calculate base addresses
    if (input) {
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
    int M = M_mat / vector_size;
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

    // Configure parameters
    int base_attn_span = 0;
    int adpbias_attn_span = 0;
    data = 0;
    data += base_attn_span;
    data += adpbias_attn_span << 23;
    iowrite32(dev, 0x1C, data);

    int num_vector = M_mat / 16;
    int num_timestep = N0;
    int adpbias_act1 = adf_accum_bias;
    if (input) {
        adpbias_act1 = input -> bias;
    }
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    iowrite32(dev, 0x20, data);

    // Configure matrix size
    data = 0x0;
    data += N0;
    data += M_mat << 20;
    iowrite32(dev, 0x10, data);

    // Set base of matrix
    data = 0;
    int base_input = 0;
    int base_output = input_buffer_size - N0 * M_mat;
    if (input == NULL) {
        base_input = input_buffer_size - N0 * M_mat;
        base_output = 0;
    }
    data = base_input;
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
    write_matrix(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        output,
        num_interrupts
    );
}

// Perform general matrix multiplication with possibility of softamx
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
    int weight_bias,
    int softmax,
    token_t *span_mask
) {
    unsigned N0_tile;
    unsigned N1_tile;

    // Try to get as many columns
    N1_tile = mask_buffer_size * bits_in_bytes / M_mat;
    N1_tile = (N1_tile / 16) * 16;
    N1_tile = min(N1_tile, N1);

    N0_tile = mask_buffer_size * bits_in_bytes / (M_mat + N1_tile);
    N0_tile = (N0_tile / 16) * 16;
    N0_tile = min(N0_tile, N0);

    // Allocate memory for matrices
    struct mat *left = aligned_malloc(sizeof(struct mat));
    token_t *val_left = aligned_malloc(N0_tile * M_mat);
    token_t *mask_left = aligned_malloc(N0_tile * M_mat / bits_in_bytes);
    int bias_left = mat1 -> bias;
    *left = (struct mat) {val_left, mask_left, bias_left};

    struct mat *right = aligned_malloc(sizeof(struct mat));
    token_t *val_right = aligned_malloc(M_mat * N1_tile);
    token_t *mask_right = aligned_malloc(M_mat * N1_tile / bits_in_bytes);
    int bias_right = mat2 -> bias;
    *right = (struct mat) {val_right, mask_right, bias_right};

    struct mat *smaller_output = aligned_malloc(sizeof(struct mat));
    token_t *val_smaller_output = aligned_malloc(N0_tile * N1_tile);
    token_t *mask_smaller_output = aligned_malloc(N0_tile * N1_tile / bits_in_bytes);
    int bias_smaller_output = 0;
    *smaller_output = (struct mat) {val_smaller_output, mask_smaller_output, bias_smaller_output};

    struct mat *output = aligned_malloc(sizeof(struct mat));
    token_t *val_output = aligned_malloc(N0 * N1);
    token_t *mask_output = aligned_malloc(N0 * N1 / bits_in_bytes);
    int bias_ouptut = 0;
    *output = (struct mat) {val_output, mask_output, bias_ouptut};

    token_t *smaller_span_mask = aligned_malloc(N0_tile * N1_tile / bits_in_bytes);

    // Tranpose for easier access
    CPU_transpose(mat2 -> values, M_mat, N1);
    CPU_transpose_mask(mat2 -> mask, M_mat, N1);

    int row = 0, col = 0;
    while (row < N0) {
        // Get left matrix
        unsigned N0_mat = min(N0_tile, N0 - row);
        memcpy(left -> values, mat1 -> values + M_mat * row, N0_mat * M_mat);
        memcpy(left -> mask, mat1 -> mask + (M_mat * row / bits_in_bytes), N0_mat * M_mat / bits_in_bytes);

        while (col < N1) {
            // Get right matrix
            unsigned N1_mat = min(N1_tile, N1 - col);
            memcpy(right -> values, mat2 -> values + M_mat * col, N1_mat * M_mat);
            memcpy(right -> mask, mat2 -> mask + (M_mat * col / bits_in_bytes), N1_mat * M_mat / bits_in_bytes);

            // Transpose back
            CPU_transpose(right -> values, N1_mat, M_mat);
            CPU_transpose_mask(right -> mask, N1_mat, M_mat);

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
                softmax,
                smaller_output
            );

            // Apply softmax only if a whole row can fit
            if (softmax && N1_mat == N1) {
                for (int i = 0; i < N0_mat; i ++) {
                    for (int j = 0; j < N1_mat / bits_in_bytes; j++) {
                        smaller_span_mask[i * N1_mat / bits_in_bytes + j] = span_mask[(i * N1) / bits_in_bytes + j];
                    }
                }

                EdgeBert_atten_softmax(
                    dev,
                    plic_dev,
                    mem,
                    N0_mat,
                    N1_mat,
                    smaller_output,
                    smaller_span_mask,
                    smaller_output
                );
            }

            // Copy over data into output
            for (int i = 0; i < N0_mat; i++) {
                for (int j = 0; j < N1_mat; j++) {
                    output -> values[N1 * (row + i) + col + j] = smaller_output -> values[N1_mat * i + j];
                }
            }

            // Copy over mask into output
            for (int i = 0; i < N0_mat; i++) {
                for (int j = 0; j < N1_mat / bits_in_bytes; j++) {
                    output -> mask[(N1 * (row + i) + col) / bits_in_bytes + j] = smaller_output -> mask[N1_mat / bits_in_bytes * i + j];
                }
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
    aligned_free(val_smaller_output);
    aligned_free(mask_smaller_output);
    aligned_free(smaller_output);
    return output;
}

// Perform softamx on matrices that cannot fit on a decoder (in-place operation)
static void general_softmax(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    token_t *span_mask
) {
    unsigned N0_tile;

    // Try to get as many rows
    N0_tile = mask_buffer_size * bits_in_bytes / (2 * N0);
    N0_tile = (N0_tile / 16) * 16;
    N0_tile = min(N0_tile, N0);

    // Allocate memory for matrices
    struct mat *input = aligned_malloc(sizeof(struct mat));
    token_t *val_input = aligned_malloc(N0_tile * M_mat);
    token_t *mask_input = aligned_malloc(N0_tile * M_mat / bits_in_bytes);
    int bias_input = mat1 -> bias;
    *input = (struct mat) {val_input, mask_input, bias_input};

    token_t *smaller_span_mask = aligned_malloc(N0_tile * M_mat / bits_in_bytes);

    int row = 0;
    while (row < N0) {
        // Get input matrix
        unsigned N0_mat = min(N0_tile, N0 - row);
        memcpy(input -> values, mat1 -> values + M_mat * row, N0_mat * M_mat);
        memcpy(input -> mask, mat1 -> mask + (M_mat * row / bits_in_bytes), N0_mat * M_mat / bits_in_bytes);

        for (int i = 0; i < N0_mat; i ++) {
            for (int j = 0; j < M_mat / bits_in_bytes; j++) {
                smaller_span_mask[i * M_mat / bits_in_bytes + j] = span_mask[(i * M_mat) / bits_in_bytes + j];
            }
        }

        // Apply softmax
        EdgeBert_atten_softmax(
            dev,
            plic_dev,
            mem,
            N0_mat,
            M_mat,
            input,
            smaller_span_mask,
            input
        );

        // Copy over data into output
        for (int i = 0; i < N0_mat; i++) {
            for (int j = 0; j < M_mat; j++) {
                mat1 -> values[M_mat * (row + i) + j] = input -> values[M_mat * i + j];
            }
        }

        // Copy over mask into output
        for (int i = 0; i < N0_mat; i++) {
            for (int j = 0; j < M_mat / bits_in_bytes; j++) {
                mat1 -> mask[(M_mat * (row + i)) / bits_in_bytes + j] = input -> mask[M_mat * i + j];
            }
        }
        row += N0_mat;
    }

    aligned_free(val_input);
    aligned_free(mask_input);
    aligned_free(input);
}

// Apply element-wise addition (return in output)
static void EdgeBert_element_add(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    struct mat *mat2,
    int write,
    struct mat *output
) {
    EdgeBert_init(dev, plic_dev, mem);
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
    // Set base output
    data = 0;
    data += input_buffer_size - N0 * M_mat;
    iowrite32(dev, 0x48, data);

    // Start element-wise addition
    data = 0xA;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Write data outside
    if (write == 0) {
        write_matrix(
            dev,
            plic_dev,
            mem,
            N0,
            M_mat,
            output,
            num_interrupts
        );
    }
}

// Apply layer norm
static void EdgeBert_layer_norm(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    int base_gamma,
    int adpbias_gamma,
    int base_beta,
    int adpbias_beta,
    struct mat *output
) {
    EdgeBert_init(dev, plic_dev, mem);
    int num_interrupts = 0;
    unsigned data = 0;

    if (mat1) {
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
    data = 0;
    data += base_gamma << 7;
    data += base_beta << 15;
    data += adpbias_gamma << 26;
    data += adpbias_beta << 29;
    iowrite32(dev, 0x1C, data);

    int num_vector = M_mat / 16;
    int num_timestep = N0;
    int adpbias_act1 = adf_accum_bias;
    if (mat1) {
        adpbias_act1 = mat1 -> bias;
    }
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    iowrite32(dev, 0x20, data);

    // Set base input
    data = 0;
    int base_input = 0;
    int base_output = input_buffer_size - N0 * M_mat;
    if (mat1 == NULL) {
        base_input = input_buffer_size - N0 * M_mat;
        base_output = 0;
    }
    data = base_input;
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
    write_matrix(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        output,
        num_interrupts
    );
}

// Performs element-wise addition with the possibility of applying layer normalizations (returns new pointer)
static struct mat *general_element_add(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    struct mat *mat2,
    int layer_norm,
    int base_gamma,
    int adpbias_gamma,
    int base_beta,
    int adpbias_beta
) {
    unsigned N0_tile;

    // Try to get as many rows
    N0_tile = mask_buffer_size * bits_in_bytes / (2 * N0);
    N0_tile = (N0_tile / 16) * 16;
    N0_tile = min(N0_tile, N0);

    // Allocate memory for matrices
    struct mat *left = aligned_malloc(sizeof(struct mat));
    token_t *val_left = aligned_malloc(N0_tile * M_mat);
    token_t *mask_left = aligned_malloc(N0_tile * M_mat / bits_in_bytes);
    int bias_left = mat1 -> bias;
    *left = (struct mat) {val_left, mask_left, bias_left};

    struct mat *right = aligned_malloc(sizeof(struct mat));
    token_t *val_right = aligned_malloc(N0_tile * M_mat);
    token_t *mask_right = aligned_malloc(N0_tile * M_mat / bits_in_bytes);
    int bias_right = mat2 -> bias;
    *right = (struct mat) {val_right, mask_right, bias_right};

    struct mat *output = aligned_malloc(sizeof(struct mat));
    token_t *val_output = aligned_malloc(N0 * M_mat);
    token_t *mask_output = aligned_malloc(N0 * M_mat / bits_in_bytes);
    int bias_ouptut = 0;
    *output = (struct mat) {val_output, mask_output, bias_ouptut};

    int row = 0;
    while (row < N0) {
        // Get left matrix
        unsigned N0_mat = min(N0_tile, N0 - row);
        memcpy(left -> values, mat1 -> values + M_mat * row, N0_mat * M_mat);
        memcpy(left -> mask, mat1 -> mask + (M_mat * row / bits_in_bytes), N0_mat * M_mat / bits_in_bytes);

        memcpy(right -> values, mat2 -> values + M_mat * row, N0_mat * M_mat);
        memcpy(right -> mask, mat2 -> mask + (M_mat * row / bits_in_bytes), N0_mat * M_mat / bits_in_bytes);

        // Add
        EdgeBert_element_add(
            dev,
            plic_dev,
            mem,
            N0_mat,
            M_mat,
            left,
            right,
            layer_norm,
            left
        );

        // Apply softmax
        if (layer_norm) {
            EdgeBert_layer_norm(
                dev,
                plic_dev,
                mem,
                N0_mat,
                M_mat,
                left,
                base_gamma,
                adpbias_gamma,
                base_beta,
                adpbias_beta,
                left
            );
        }

        // Copy over data into output
        for (int i = 0; i < N0_mat; i++) {
            for (int j = 0; j < M_mat; j++) {
                output -> values[M_mat * (row + i) + j] = left -> values[M_mat * i + j];
            }
        }

        // Copy over mask into output
        for (int i = 0; i < N0_mat; i++) {
            for (int j = 0; j < M_mat / bits_in_bytes; j++) {
                output -> mask[(M_mat * (row + i)) / bits_in_bytes + j] = left -> mask[M_mat * i + j];
            }
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

// Performs layer norm for matrices that cannot fit in decoder (in-place operation)
static void general_layer_norm(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int N0,
    int M_mat,
    struct mat *mat1,
    int base_gamma,
    int adpbias_gamma,
    int base_beta,
    int adpbias_beta
) {
    unsigned N0_tile;

    // Try to get as many rows
    N0_tile = mask_buffer_size * bits_in_bytes / (2 * N0);
    N0_tile = (N0_tile / 16) * 16;
    N0_tile = min(N0_tile, N0);

    // Allocate memory for matrices
    struct mat *input = aligned_malloc(sizeof(struct mat));
    token_t *val_input = aligned_malloc(N0_tile * M_mat);
    token_t *mask_input = aligned_malloc(N0_tile * M_mat / bits_in_bytes);
    int bias_input = mat1 -> bias;
    *input = (struct mat) {val_input, mask_input, bias_input};

    int row = 0;
    while (row < N0) {
        // Get input matrix
        unsigned N0_mat = min(N0_tile, N0 - row);
        memcpy(input -> values, mat1 -> values + M_mat * row, N0_mat * M_mat);
        memcpy(input -> mask, mat1 -> mask + (M_mat * row / bits_in_bytes), N0_mat * M_mat / bits_in_bytes);

        // Add
        EdgeBert_layer_norm(
            dev,
            plic_dev,
            mem,
            N0_mat,
            M_mat,
            input,
            base_gamma,
            adpbias_gamma,
            base_beta,
            adpbias_beta,
            input
        );

        // Copy over data into output
        for (int i = 0; i < N0_mat; i++) {
            for (int j = 0; j < M_mat; j++) {
                mat1 -> values[M_mat * (row + i) + j] = input -> values[M_mat * i + j];
            }
        }

        // Copy over mask into output
        for (int i = 0; i < N0_mat; i++) {
            for (int j = 0; j < M_mat / bits_in_bytes; j++) {
                mat1 -> mask[(M_mat * (row + i)) / bits_in_bytes + j] = input -> mask[M_mat * i + j];
            }
        }
        row += N0_mat;
    }

    aligned_free(val_input);
    aligned_free(mask_input);
    aligned_free(input);
}

static void EdgeBert_entropy(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem
) {
    // TODO
    return;
}

static struct mat *EdgeBert_pooler(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    struct mat *attention_heads,
    int input_m,
    int hidden_size
) {
    // Weight matrix (hidden_size * hidden_size)
    struct mat *we_mat1 = aligned_malloc(sizeof(struct mat));
    token_t *val_mat1 = aligned_malloc(hidden_size * hidden_size);
    token_t *mask_mat1 = aligned_malloc(hidden_size * hidden_size / bits_in_bytes);
    int bias_mat1 = 0;
    *we_mat1 = (struct mat) {val_mat1, mask_mat1, bias_mat1};

    struct mat *input = aligned_malloc(sizeof(struct mat));
    token_t *val_input = aligned_malloc(hidden_size);
    token_t *mask_input = aligned_malloc(hidden_size / bits_in_bytes);
    *input = (struct mat) {val_input, mask_input, attention_heads -> bias};

    // Fill with data
    // EdgeBert_init_buf_pooler(val_mat1, mask_mat1, bias_mat1);
    memset(val_mat1, 1, hidden_size * hidden_size);
    memset(mask_mat1, 255, hidden_size * hidden_size / bits_in_bytes);

    // Take first token
    memcpy(val_input, attention_heads -> values, hidden_size);
    memcpy(mask_input, attention_heads -> mask, hidden_size / bits_in_bytes);

    // Matrix multiplication configurations
    unsigned N0 = input_m;
    unsigned N1 = hidden_size;
    unsigned M_mat = 1;
    unsigned is_relu = 0;
    unsigned is_bias = 0;
    unsigned weight_bias = 0;
    unsigned softmax = 0;

    // Query multiplication
    struct mat* output = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        we_mat1,
        input,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    // Activation?

    // Free allocated space
    aligned_free(val_mat1);
    aligned_free(mask_mat1);
    aligned_free(we_mat1);

    aligned_free(val_input);
    aligned_free(mask_input);
    aligned_free(input);
    return output;
}

static struct mat *EdgeBert_highway_exit(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    struct mat *attention_heads,
    int input_m,
    int hidden_size,
    int num_labels
) {
    // Highway exit
    // Index in to layer outputs
    struct mat *pooler_output = EdgeBert_pooler(
        dev,
        plic_dev,
        mem,
        attention_heads,
        input_m,
        hidden_size
    );

    // Intialize linear layer and outputs
    struct mat *we_mat1 = aligned_malloc(sizeof(struct mat));
    token_t *val_mat1 = aligned_malloc(num_labels * hidden_size);
    token_t *mask_mat1 = aligned_malloc(num_labels * hidden_size / bits_in_bytes);
    int bias_mat1 = 0;
    *we_mat1 = (struct mat) {val_mat1, mask_mat1, bias_mat1};

    // Fill with data
    // EdgeBert_init_buf_highway(we_mat1);
    memset(val_mat1, 1, num_labels * hidden_size);
    memset(mask_mat1, 255, num_labels * hidden_size / bits_in_bytes);

    // Initialize config
    unsigned N0 = num_labels;
    unsigned N1 = 1;
    unsigned M_mat = hidden_size;
    unsigned is_relu = 0;
    unsigned is_bias = 0;
    unsigned weight_bias = 0;
    unsigned softmax = 0;

    // Perform matrix multiplication
    struct mat* output = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        we_mat1,
        pooler_output,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    // Free memory
    aligned_free(val_mat1);
    aligned_free(mask_mat1);
    aligned_free(we_mat1);
    aligned_free(pooler_output -> values);
    aligned_free(pooler_output -> mask);
    aligned_free(pooler_output);

    return output;
}

// Attention head
static struct mat *EdgeBert_attention(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    struct mat* input,
    int input_m,
    int input_n,
    int hidden_size
) {
    printf("STARTing Attention Head in EdgeBert...\n");
    int num_interrupts = 0;

    struct mat *we_query = aligned_malloc(sizeof(struct mat));
    token_t *val_query = aligned_malloc(input_n * hidden_size);
    token_t *mask_query = aligned_malloc(input_n * hidden_size / bits_in_bytes);
    int bias_query = 0;
    *we_query = (struct mat) {val_query, mask_query, bias_query};

    struct mat *we_key = aligned_malloc(sizeof(struct mat));
    token_t *val_key = aligned_malloc(input_n * hidden_size);
    token_t *mask_key = aligned_malloc(input_n * hidden_size / bits_in_bytes);
    int bias_key = 0;
    *we_key = (struct mat) {val_key, mask_key, bias_key};

    struct mat *we_val = aligned_malloc(sizeof(struct mat));
    token_t *val_val = aligned_malloc(input_n * hidden_size);
    token_t *mask_val = aligned_malloc(input_n * hidden_size / bits_in_bytes);
    int bias_val = 0;
    *we_val = (struct mat) {val_val, mask_val, bias_val};

    token_t *span_mat = aligned_malloc(input_m * input_m / bits_in_bytes);

    // Initialize weights
    // EdgeBert_init_buf_attention(
    //     input_ids,
    //     mask_input,
    //     we_query,
    //     mask_query,
    //     we_key,
    //     mask_key,
    //     we_val,
    //     mask_val,
    //     aux_mat
    // );
    memset(val_query, 35, input_n * hidden_size);
    memset(mask_query, 255, input_n * hidden_size / bits_in_bytes);
    memset(val_key, -1, input_n * hidden_size);
    memset(mask_key, 255, input_n * hidden_size / bits_in_bytes);
    memset(val_val, -12, input_n * hidden_size);
    memset(mask_val, 255, input_n * hidden_size / bits_in_bytes);
    memset(span_mat, 255, input_m * input_m / bits_in_bytes);

    // Initialize matrix multiplication
    unsigned N0 = input_m;
    unsigned N1 = hidden_size;
    unsigned M_mat = input_n;
    unsigned is_relu = 0;
    unsigned is_bias = 0;
    unsigned weight_bias = 0;
    unsigned softmax = 0;

    // Mutliply IDs by query matrix
    struct mat *mat_query = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        input,
        we_query,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    // Mutliply IDs by key matrix
    struct mat *mat_key = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        input,
        we_key,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    // Mutliply IDs by value matrix
    struct mat *mat_val = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        input,
        we_val,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    // Free inputs and weights
    aligned_free(input -> values);
    aligned_free(input -> mask);
    aligned_free(input);

    aligned_free(val_query);
    aligned_free(mask_query);
    aligned_free(we_query);

    aligned_free(val_key);
    aligned_free(mask_key);
    aligned_free(we_key);

    aligned_free(val_val);
    aligned_free(mask_val);
    aligned_free(we_val);

    // Transpose output of key multiplication
    CPU_transpose(mat_key -> values, N0, N1);
    CPU_transpose_mask(mat_key -> mask, N0, N1);

    // Matrix config
    N0 = input_m;
    M_mat = input_n;
    N1 = input_m;
    // Set softmax parameter to true
    softmax = 1;

    // Multiply query and key output
    struct mat *mat_query_key;
    if (mask_buffer_size * bits_in_bytes / M_mat >= N1) {
        mat_query_key = general_mat_mul(
            dev,
            plic_dev,
            mem,
            N0,
            N1,
            M_mat,
            mat_query,
            mat_key,
            is_relu,
            is_bias,
            weight_bias,
            softmax,
            span_mat
        );
    } else {
        softmax = 0;
        mat_query_key = general_mat_mul(
            dev,
            plic_dev,
            mem,
            N0,
            N1,
            M_mat,
            mat_query,
            mat_key,
            is_relu,
            is_bias,
            weight_bias,
            softmax,
            span_mat
        );

        general_softmax(
            dev,
            plic_dev,
            mem,
            N0,
            N1,
            mat_query_key,
            span_mat
        );
    }

    // Free memory
    aligned_free(mat_query -> values);
    aligned_free(mat_query -> mask);
    aligned_free(mat_query);
    aligned_free(mat_key -> values);
    aligned_free(mat_key -> mask);
    aligned_free(mat_key);
    aligned_free(span_mat);

    // Multiply query and key with value matrix
    N0 = input_m;
    M_mat = input_m;
    N1 = hidden_size;
    softmax = 0;

    struct mat *output = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        mat_query_key,
        mat_val,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    // Free memory
    aligned_free(mat_val -> values);
    aligned_free(mat_val -> mask);
    aligned_free(mat_val);

    aligned_free(mat_query_key -> values);
    aligned_free(mat_query_key -> mask);
    aligned_free(mat_query_key);

    printf("FINISHing Attention Head in EdgeBert...\n");
    return output;
}

static struct mat *EdgeBert_attention_heads(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    struct mat *input,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels
) {
    printf("STARTing EdgeBERT %d Attention Heads Computation...\n", num_heads);

    // Profiling
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    struct mat *attention_heads = aligned_malloc(sizeof(struct mat));
    token_t *val_attention_heads = aligned_malloc(input_m * hidden_size * num_heads);
    token_t *mask_attention_heads = aligned_malloc(input_m * hidden_size * num_heads / bits_in_bytes);
    *attention_heads = (struct mat) {val_attention_heads, mask_attention_heads, adf_accum_bias};

    for (int i = 0; i < 12; i++) {
        // Run attention head
        count1 = get_counter();
        struct mat *head_output = EdgeBert_attention(
            dev,
            plic_dev,
            mem,
            input,
            input_m,
            input_n,
            hidden_size
        );
        count2 = get_counter();
        exe_cycle = count2 - count1;
        printf("...Attention Head %d takes %"PRIu64" clock cycles...\n", i + 1, exe_cycle);

        // Copy over values
        for (int j = 0; j < input_m; j++) {
            for (int k = 0; k < hidden_size; k++) {
                attention_heads -> values[hidden_size * num_heads * j + hidden_size * i + k] = head_output -> values[hidden_size * j + k];
            }
        }

        // Copy over mask
        for (int j = 0; j < input_m; j++) {
            for (int k = 0; k < hidden_size / bits_in_bytes; k++) {
                attention_heads -> mask[(hidden_size * num_heads * j + hidden_size * i) / bits_in_bytes + k] = head_output -> mask[hidden_size / bits_in_bytes * j + k];
            }
        }

        // Keep track of number of cycles
        total_exe_cycle = total_exe_cycle + exe_cycle;
    }

    printf("FINISHing EdgeBERT 12 Attention Heads Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
    return attention_heads;
}

static struct mat *EdgeBert_processing(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    struct mat *attention_heads,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size
) {
    printf("STARTing 12 Attention Heads Processing...\n");

    // Profiling
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    count1 = get_counter();
    struct mat *we_mat1 = aligned_malloc(sizeof(struct mat));
    token_t* val_mat1 = aligned_malloc(num_heads * hidden_size * input_n);
    token_t *mask_mat1 = aligned_malloc(num_heads * hidden_size * input_n / bits_in_bytes);
    int bias_mat1 = 0;
    *we_mat1 = (struct mat) {val_mat1, mask_mat1, bias_mat1};

    // Fill with dummy data
    // EdgeBert_init_buf_processing(we_mat1);
    memset(val_mat1, 1, num_heads * hidden_size * input_n);
    memset(mask_mat1, 255, num_heads * hidden_size * input_n / bits_in_bytes);

    // Matrix multiplication configurations
    unsigned N0 = input_m;
    unsigned N1 = input_n;
    unsigned M_mat = hidden_size * num_heads;
    unsigned is_relu = 0;
    unsigned is_bias = 0;
    unsigned weight_bias = 0;
    unsigned softmax = 0;

    struct mat *mat_output1 = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        attention_heads,
        we_mat1,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    // Free memory
    aligned_free(attention_heads -> values);
    aligned_free(attention_heads -> mask);
    aligned_free(attention_heads);

    aligned_free(we_mat1 -> values);
    aligned_free(we_mat1 -> mask);
    aligned_free(we_mat1);

    // Add on input IDs and layer norm
    struct mat *input = aligned_malloc(sizeof(struct mat));
    token_t *val_input = aligned_malloc(input_m * input_n);
    token_t *mask_input = aligned_malloc(input_m * input_n / bits_in_bytes);
    int bias_input = 0;
    *input = (struct mat) {val_input, mask_input, bias_input};

    memset(val_input, 11, input_m * input_n);
    memset(mask_input, 255, input_m * input_n / bits_in_bytes);

    N0 = input_m, M_mat = input_n;

    // Add and layer norm
    unsigned layer_norm = 1;
    struct mat *output = general_element_add(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        input,
        mat_output1,
        layer_norm,
        1,
        0,
        0,
        0
    );
    count2 = get_counter();
    printf("FINISHing 12 Attention Heads Processing...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);

    return output;
}

// Feed forward
static struct mat *EdgeBert_feed_forward(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    struct mat *attention_head_out,
    int input_m,
    int input_n,
    int hidden_size_ffn
) {
    printf("STARTing EdgeBERT Feed Forward Net Computation...\n");
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    count1 = get_counter();
    // Initialize weights
    struct mat *we_mat1 = aligned_malloc(sizeof(struct mat));
    token_t *val_mat1 = aligned_malloc(input_n * hidden_size_ffn);
    token_t *mask_mat1 = aligned_malloc(input_n * hidden_size_ffn / bits_in_bytes);
    int bias_mat1 = 0;
    *we_mat1 = (struct mat) {val_mat1, mask_mat1, bias_mat1};

    struct mat *we_mat2 = aligned_malloc(sizeof(struct mat));
    token_t *val_mat2 = aligned_malloc(input_n * hidden_size_ffn);
    token_t *mask_mat2 = aligned_malloc(input_n * hidden_size_ffn / bits_in_bytes);
    int bias_mat2 = 0;
    *we_mat2 = (struct mat) {val_mat2, mask_mat2, bias_mat2};

    // Load data
    // init_buf_ffn(we_mat1, we_mat2);
    memset(val_mat1, -1, input_n * hidden_size_ffn);
    memset(val_mat2, -12, input_n * hidden_size_ffn);
    memset(mask_mat1, 255, input_n * hidden_size_ffn / bits_in_bytes);
    memset(mask_mat2, 255, input_n * hidden_size_ffn / bits_in_bytes);

    // Multiply attention output by weights
    unsigned N0 = input_m;
    unsigned N1 = hidden_size_ffn;
    unsigned M_mat = input_n;
    unsigned is_relu = 0;
    unsigned is_bias = 0;
    unsigned weight_bias = 0;
    unsigned softmax = 0;

    struct mat *mat_output1 = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        attention_head_out,
        we_mat1,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    aligned_free(we_mat1 -> values);
    aligned_free(we_mat1 -> mask);
    aligned_free(we_mat1);

    // Multiply with second weights
    N0 = input_m;
    M_mat = hidden_size_ffn;
    N1 = input_n;

    struct mat *mat_output2 = general_mat_mul(
        dev,
        plic_dev,
        mem,
        N0,
        N1,
        M_mat,
        mat_output1,
        we_mat2,
        is_relu,
        is_bias,
        weight_bias,
        softmax,
        NULL
    );

    N0 = input_m;
    M_mat = input_n;

    // Add attention output
    unsigned layer_norm = 1;
    struct mat *output = general_element_add(
        dev,
        plic_dev,
        mem,
        N0,
        M_mat,
        attention_head_out,
        mat_output2,
        layer_norm,
        1,
        0,
        0,
        0
    );

    // Free memory
    aligned_free(attention_head_out -> values);
    aligned_free(attention_head_out -> mask);
    aligned_free(attention_head_out);

    aligned_free(we_mat2 -> values);
    aligned_free(we_mat2 -> mask);
    aligned_free(we_mat2);

    count2 = get_counter();
    printf("FINISHing EdgeBERT Feed Forward Net Computation...\n");
    printf("###(taking %"PRIu64" clock cycles)###...\n", count2 - count1);
    return output;
}

static struct mat *EdgeBert_transformer(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    struct mat *input,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels,
    int hidden_size_ffn
) {
    printf("STARTing EdgeBERT Transformer Layer...\n");
    // Attention heads
    struct mat *attention_heads = EdgeBert_attention_heads(
        dev,
        plic_dev,
        mem,
        input,
        num_heads,
        input_m,
        input_n,
        hidden_size,
        num_labels
    );

    // Processing
    printf("\n");
    struct mat *attention_head_out = EdgeBert_processing(
        dev,
        plic_dev,
        mem,
        attention_heads,
        num_heads,
        input_m,
        input_n,
        hidden_size
    );

    // Feed Forward Neural Net
    printf("\n");
    struct mat *out = EdgeBert_feed_forward(
        dev,
        plic_dev,
        mem,
        attention_head_out,
        input_m,
        input_n,
        hidden_size_ffn
    );

    printf("FINISHing EdgeBERT Transformer Layer...\n");
    return out;
}

static void EdgeBert_transformer_layers(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int num_layers,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int num_labels,
    int hidden_size_ffn
) {
    printf("Transformer Matmul Performance on EdgeBERT BEGIN...\n");
    // Initialize inputs
    struct mat *input = aligned_malloc(sizeof(struct mat));
    token_t *val_input = aligned_malloc(input_m * input_n);
    token_t *mask_input = aligned_malloc(input_m * input_n / bits_in_bytes);
    int bias_input = 0;
    *input = (struct mat) {val_input, mask_input, bias_input};

    memset(val_input, 11, input_m * input_n);
    memset(mask_input, 255, input_m * input_n / bits_in_bytes);

    for (int i = 0; i < num_layers; i++) {
        input = EdgeBert_transformer(
            dev,
            plic_dev,
            mem,
            input,
            num_heads,
            input_m,
            input_n,
            hidden_size,
            num_labels,
            hidden_size_ffn
        );
    }

    aligned_free(input -> values);
    aligned_free(input -> mask);
    aligned_free(input);

    printf("Transformer Matmul Performance on EdgeBERT FINISH...\n");
    printf("Thank you!\n");
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
    CPU_transformer_layers(
        1,
        12,
        128,
        768,
        64,
        2,
        3072
    );

    // Run transformer on accelerator
    EdgeBert_transformer_layers(
        &dev,
        &plic_dev,
        mem,
        1,
        12,
        128,
        768,
        64,
        2,
        3072
    );

    printf("FINISHing DRIVER\n");
    aligned_free(mem);
    return 0;
}
