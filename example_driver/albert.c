/* Copyright (c) 2011-2021 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */

#include "albert.h"
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

// Get direct memory access (DMA) per beat
static unsigned DMA_WORD_PER_BEAT(unsigned _st) {
    return (sizeof(void *) / _st);
}

// Config
// Constants
static unsigned mem_size;
const static int N = 16;
const static int M = N;

static unsigned is_relu;
const static unsigned is_bias = 1;
const static int weight_bias = -8;
const static int adf_accum_bias = 2;
const static int accum_right_shift = 2;

// Attention span
const static int base_attn_span = 0;
// Layer normalization
const static int base_gamma = 8;
const static int base_beta = 56;
// Adaptiv float bias for parameters
const static int adpbias_attn_span = 2;
const static int adpbias_gamma = 2;
const static int adpbias_beta = 2;

const static int num_vector = 8;
const static int num_timestep = 128;

// QUESTION: What are these?/Why are there three?
const static int adpbias_act1 = 2;
const static int adpbias_act2 = 2;
const static int adpbias_act3 = 2;

// Base output to store computations
const static int base_output = 1024;
const static int base_input0 = 0;
const static int base_input1 = 0;

// Matrix configurations
const static unsigned N0 = 16;
const static unsigned N1 = 16;
const static unsigned M_mat = 16;

// Size of mask buffer (in bytes)
const static unsigned mask_buffer_size = 8192;
// Size of data buffer (in bytes)
const static unsigned input_buffer_size = 65536;
// Size of aux buffer (in bytes)
const static unsigned aux_buffer_size = 4096;

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


// CPU functions
// Helper functions
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
void CPU_softmax(float* input, size_t size) {
    // Scale down to prevent overflow
    int i;
    float m, sum, constant;

    // Find max of array
    m = -INFINITY;
    for (i = 0; i < size; i++) {
        if (m < input[i]) {
            m = input[i];
        }
    }

    // Subtract off largets element and exp
    sum = 0.0;
    for (i = 0; i < size; i++) {
        sum += exp(input[i] - m);
    }

    // Subtract off max and put sum in denominator
    constant = m + log(sum);
    for (i = 0; i < size; i++) {
        input[i] = exp(input[i] - constant);
    }
}

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
            if (((mask[mask_idx] >> (8 - offset - 1)) & 1) == 1) {
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
                mask[mask_idx] += (1 << (8 - offset - 1));
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

// Calculate entropy
float *CPU_entropy(float *input, int m, int n) {
    float e[m * n];
    float e_x[m * n];
    // Apply exp to each entry
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            e[i * n + j] = exp(input[i * n + j]);
            e_x[i * n + j] = input[i * n + j] * exp(input[i * n + j]);;
        }
    }

    // Sum over rows
    float a[m];
    float b[m];
    for (int i = 0; i < m; i++) {
        float sum_a = 0.0;
        float sum_b = 0.0;
        for (int j = 0; j < n; j++) {
            sum_a += e[i * n + j];
            sum_b += e_x[i * n + j];
        }
        a[i] = sum_a;
        b[i] = sum_b;
    }

    // Calculate entropy
    float *out = malloc(m * sizeof(float));
    for (int i = 0; i < m; i++) {
        out[i] = log(a[i]) - b[i] / a[i];
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
    int *we;
    int *output;

    // Allocate space for matrices
    we = aligned_malloc(hidden_size * hidden_size * sizeof(int));
    output = aligned_malloc(input_m * hidden_size * sizeof(int));

    // Fill with dummy data
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        we[i] = 10;
    }

    // Matrix multiplication configurations
    int N0;
    int N1;
    int M_mat;
    N0 = input_m; M_mat = hidden_size; N1 = hidden_size;

    // Query multiplication
    CPU_multiply(we, attention_heads, N0, M_mat, N1, output);

    // Activation?

    // Get hidden states for first token
    int *out;
    out = aligned_malloc(hidden_size * sizeof(int));
    memcpy(out, output, hidden_size * sizeof(int));

    aligned_free(we);
    aligned_free(output);

    return out;
}

// Highway Exit
static int *CPU_EdgeBert_highway_exit(
    int *attention_heads,
    int input_m,
    int hidden_size,
    int he_layer1,
    int he_layer2
) {
    int *pooler_output;
    pooler_output = CPU_EdgeBert_pooler(attention_heads, input_m, hidden_size);

    int *we_mat1;
    int *we_mat2;
    int *output1;
    int *output2;

    we_mat1 = aligned_malloc(he_layer1 * hidden_size * sizeof(int));
    we_mat1 = aligned_malloc(he_layer2 * he_layer1 * sizeof(int));
    output1 = aligned_malloc(he_layer1 * sizeof(int));
    output2 = aligned_malloc(he_layer2 * sizeof(int));

    // Fill with dummy data
    for (int i = 0; i < he_layer1 * hidden_size; i++) {
        we_mat1[i] = 4;
    }
    for (int i = 0; i < he_layer2 * he_layer1; i++) {
        we_mat2[i] = 2;
    }

    int N0;
    int N1;
    int M_mat;
    N0 = he_layer1; M_mat = hidden_size; N1 = 1;
    CPU_multiply(we_mat1, pooler_output, N0, M_mat, N1, output1);
    N0 = he_layer2; M_mat = he_layer1; N1 = 1;
    CPU_multiply(we_mat2, output1, N0, M_mat, N1, output2);

    // Entropy?
    float entropy = CPU_entropy(output2, he_layer2, 1);
    float threshold = 0.0;
    if (entropy < threshold) {
        return output2;
    }
    return NULL;
}

// Attention head
static int *CPU_EdgeBert_attention(
    int input_m,
    int input_n,
    int hidden_size,
    int he_layer1,
    int he_layer2
) {
    printf("STARTing Attention Head in CPU...\n");
    // Length of 128 tokens, each token has 768 entries
    int *input_ids;
    // Query, key, and value matrices (768 x 64)
    int *we_query;
    int *we_key;
    int *we_val;
    // Output of multiplication (128 x 64)
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

    // Fill with dummy data
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

    // Query output (128 x 64) multiplied by transpose of key output (64 x 128)
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
    aligned_free(output5);
    printf("FINISHing Attention Head in CPU...\n");
}

static void CPU_EdgeBert_attention_heads(
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int he_layer1,
    int he_layer2
) {
    printf("STARTing CPU 12 Attention Heads Computation...\n");

    // Initialize input ids
    int *attention_heads_cpu;
    attention_heads_cpu = aligned_malloc(128 * 768 * sizeof(int));

    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;
    total_exe_cycle = 0;

    // Run attention heads
    for (int i = 0; i < 12; i++) {
        // Get time for each head
        count1 = get_counter();
        int *out = CPU_EdgeBert_attention(
            input_m,
            input_n,
            hidden_size,
            he_layer1,
            he_layer2
        );
        count2 = get_counter();
        exe_cycle = count2 - count1;
        printf("...Attention Head %d takes %"PRIu64" clock cycles...\n", i, exe_cycle);
        total_exe_cycle = total_exe_cycle + exe_cycle;

        // Fill output with dummy data
        for (int l = 0; l < input_m; l++) {
            for (int k = 0; k < hidden_size; k++) {
                attention_heads_cpu[l * hidden_size * num_heads + i * hidden_size + k] = l * hidden_size + k;
            }
        }

        // Highway exit
        int *out = CPU_EdgeBert_highway_exit(out, input_m, hidden_size, he_layer1, he_layer2);
        if (out) {
            printf("FINISHing CPU 12 Attention Heads Computation EARLY...\n");
            printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
            return;
        }
    }

    printf("FINISHing CPU 12 Attention Heads Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
}

static void CPU_EdgeBert_processing(
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size
) {
    printf("STARTing CPU 12 Attention Heads Processing...\n");
    uint64_t count1;
    uint64_t count2;
    count1 = get_counter();

    // Initialize weights for processing
    int *attention_heads;
    attention_heads = aligned_malloc(input_m * hidden_size * num_heads * sizeof(int));
    int *we_heads;
    we_heads = aligned_malloc(hidden_size * num_heads * input_n * sizeof(int));
    int *attention_heads_out;
    attention_heads = aligned_malloc(input_m * input_n * sizeof(int));

    // Fill weight matrix with dummy data
    for (int i = 0; i < hidden_size * num_heads * input_n; i++) {
        we_heads[i] = -1;
    }

    int N0;
    int N1;
    int M_mat;
    N0 = input_m; M_mat = hidden_size * num_heads; N1 = input_n;
    CPU_multiply(attention_heads, we_heads, N0, M_mat, N1, attention_head_out_cpu);

    // Layer normalization?

    count2 = get_counter();
    printf("FINISHing CPU 12 Attention Heads Processing...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);
}

// Feed forward neural network after attention heads
static void CPU_EdgeBert_feed_forward(
    int input_m,
    int input_n,
    int hidden_size_ffn,
) {
    printf("STARTing CPU Feed Forward Net Computation...\n");
    uint64_t count1;
    uint64_t count2;
    count1 = get_counter();

    // Initialize input, weights, and outputs
    int *attention_head_out;
    int *we1;
    int *we2;
    int *output1;
    int *output2;
    attention_head_out = aligned_malloc(input_m * input_n * sizeof(int));
    we_mat1 = aligned_malloc(input_n * hidden_size_ffn * sizeof(int));
    we_mat2 = aligned_malloc(hidden_size_ffn * input_n * sizeof(int));
    output1 = aligned_malloc(input_m * hidden_size_ffn * sizeof(int));
    output2 = aligned_malloc(input_m * input_n * sizeof(int));

    // Fill with dummy data
    for (int i = 0; i < input_m * input_n; i++) {
        attention_head_out[i] = 38;
    }
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
    CPU_multiply(attention_head_out, we1, N0, M_mat, N1, output1);

    // Activation function?

    // Second multiplication
    N0 = input_m; M_mat = hidden_size_ffn; N1 = input_n;
    CPU_multiply(output1, we2, N0, M_mat, N1, output2);

    // Layer normalization?

    count2 = get_counter();
    printf("FINISHing CPU Feed Forward Net Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);
}

// Profile CPU performance
static void CPU_transformer() {
    printf("\nTransformer Matmul Performance Profiling on Ariane RISC-V CPU BEGIN...\n");
    CPU_EdgeBert_attention_heads();
    printf("\n")
    CPU_EdgeBert_processing();
    printf("\n");
    CPU_EdgeBert_feed_forward();
    printf("Transformer Matmul Performance Profiling on Ariane RISC-V CPU DONE...\n");
    printf("Thank you!\n");
}


// Accelerator functions
// Helper functions
static int wait(struct esp_device *plic_dev, int num_interrupts) {
    // printf("......waiting for interrupt #%i\n", num_interrupts);
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving for interrupt #%i\n", num_interrupts);
    return num_interrupts + 1;
}

// Input initialization
static void init_buf_attention(token_t *input_ids, token_t *we_query,token_t *we_key, token_t *we_val, token_t *mask_mat, token_t *aux_mat) {
    // #include "data/attention/input_ids.h"      // 128 * 768
    // #include "data/attention/head_we_query.h"  // 768 * 64
    // #include "data/attention/head_we_key.h"    // 768 * 64
    // #include "data/attention/head_we_val.h"    // 768 * 64
    // #include "data/attention/mask_mat.h"       // 8192 chars (QUESTION)
    // #include "data/attention/aux_mat.h"        // 4096 chars (QUESTION)
}

static void init_buf_processing(token_t *we_mat1) {
    // #include "data/processing/we_mat1.h"       // 768 * 768
}

static void init_buf_ffn(token_t *we_mat1, token_t *we_mat2) {
    // #include "data/ffn/we_mat1.h"              // 768 * 3072
    // #include "data/ffn/we_mat1.h"              // 3072 * 768
}

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

// Core functions
// Read for mask (decoder 0 and 1) and aux
static int EdgeBert_init(struct esp_device *dev, struct esp_device *plic_dev, token_t *mem) {
    // TODO: Our reset hack
    iowrite32(dev, 0x00, 1);

    unsigned data = 0;
    int num_interrupts = 0;

    printf("...STARTing base addr setting for EdgeBert...\n");

    // Calculate base addresses
    // Base address
    unsigned mask_rd_base;
    unsigned input_rd1_base;
    unsigned input_rd2_base;
    unsigned aux_rd_base;
    unsigned input_wr_base;
    unsigned mask_wr_base;

    mask_rd_base = ((unsigned) mem);
    input_rd1_base = ((unsigned) mem) + mask_buffer_size;
    input_rd2_base = ((unsigned) mem) + mask_buffer_size + input_buffer_size;
    aux_rd_base = ((unsigned) mem) + mask_buffer_size + 2 * input_buffer_size;
    input_wr_base = ((unsigned) mem) + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size;
    mask_wr_base = ((unsigned) mem) + mask_buffer_size + 3 * input_buffer_size + aux_buffer_size;

    data = 0;
    // Master mask read (load from outside and store in accelerator MASK scratchpad) for decoder 0
    // Store base output of computations
    data += base_output;
    iowrite32(dev, 0x48, data);
    // Not using SFU
    data = 0;
    iowrite32(dev, 0x50, data);
    // Set mode to N0
    data = 0x81;
    iowrite32(dev, 0x58, data);
    // Use 8-bit MAC
    data = 0;
    iowrite32(dev, 0x60, data);
    // Set num words to read for Input/Mask AXI
    data = 0;
    data += (M - 1);
    iowrite32(dev, 0x40, data);
    // Set mask read address
    data = mask_rd_base;
    iowrite32(dev, 0x28, data);
    // Set mask write address
    data = mask_wr_base;
    iowrite32(dev, 0x2C, data);
    // Set input write address
    data = input_wr_base;
    iowrite32(dev, 0x34, data);
    // Select decoder 0
    data = 0x0;
    iowrite32(dev, 0x08, data);
    // Start master mask read
    data = 0x01;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Master mask read (load from outside and store in accelerator MASK scratchpad) for decoder 1
    // Select decoder 1
    data = 0x1;
    iowrite32(dev, 0x08, data);
    // Start master mask read
    data = 0x01;
    iowrite32(dev,0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Start master aux read (load from outside and store in accelerator AUX scratchpad)
    // Set num words to read/write Aux AXI
    data = 0;
    data += (M - 1);
    iowrite32(dev, 0x44, data);
    // Set aux read base
    data = aux_rd_base;
    iowrite32(dev, 0x38, data);
    // Start master aux read
    data = 0x05;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    printf("...FINISHing base addr setting for EdgeBert...\n");
}

// Matrix multiplication
static int EdgeBert_mat_mul(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    int N0,
    int N1,
    int M_mat,
    int is_relu,
    token_t *mem,
    token_t *mask_mat,
    token_t *D_mat1,
    token_t *D_mat2,
    int softmax
) {
    // TODO: Load in masks
    printf("...STARTing Matmul in EdgeBert...\n");
    int num_interrupts = 0;

    unsigned data = 0;
    unsigned input_rd1_base = ((unsigned) mem) + mask_buffer_size;
    unsigned input_rd2_base = ((unsigned) mem) + mask_buffer_size + input_buffer_size;

    // Loads in matrices from memory
    memcpy(mem, mask_mat, mask_buffer_size * sizeof(token_t));
    memcpy(mem + mask_buffer_size, D_mat1, N0 * M_mat * sizeof(token_t));
    memcpy(mem + mask_buffer_size + input_buffer_size, D_mat2, M_mat * N1 * sizeof(token_t));
    // memcpy(mem + mask_buffer_size + 2 * input_buffer_size, aux_mat, aux_buffer_size * sizeof(token_t));

    // Load in matrices to accelerator
    // QUESTION
    // EdgeBert_init(dev, plic_dev, mem);
    // Set post-processing configuration
    data = 0;
    data += is_relu;
    data += is_bias << 4;
    data += weight_bias << 8;
    data += adf_accum_bias << 16;
    data += accum_right_shift << 20;
    iowrite32(dev, 0x0C, data);

    // Load in data to decoder 0
    // Start with D_mat1 in decoder 0
    data = 0x0;
    iowrite32(dev, 0x4C, data);
    iowrite32(dev, 0x08, data);
    // Set base read address
    data = input_rd1_base;
    iowrite32(dev, 0x30, data);
    // Start master input read
    data = 0x03;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Load in data to decoder 1
    // D_mat2 in decoder 1
    // Switch decoder
    data = 0x1;
    iowrite32(dev, 0x08, data);
    // Set base read address
    data = input_rd2_base;
    iowrite32(dev, 0x30, data);
    // Start master input read
    data = 0x03;
    iowrite32(dev, 0x04, data);
    num_interrupts = wait(plic_dev, num_interrupts);

    // Do matrix multiplication
    // Start matrix size configurations
    data = 0x0;
    data += N0;
    data += N1 << 10;
    data += M_mat << 20;
    iowrite32(dev, 0x10, data);
    // Set base input of first and second matrix
    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);
    // Start matrix multiplication
    data = 0x07;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Write data outside
    if (softmax == 0) {
        // Set use_axi to 1
        data = 0x1;
        iowrite32(dev, 0x4C, data);
        // Start input master write
        data = 0x04;
        iowrite32(dev, 0x04, data);
        // Wait for interrupt
        num_interrupts = wait(plic_dev, num_interrupts);
    }

    printf("...FINISHing Matmul in EdgeBert...\n");
    return num_interrupts;
}

static void general_mat_mul(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    int N0,
    int N1,
    int M_mat,
    int is_relu,
    token_t *mem,
    token_t *mask_mat,
    token_t *D_mat1,
    token_t *D_mat2,
    int softmax
) {
    // TODO: Deal with masks
    unsigned N0_tile;
    unsigned N1_tile;

    // Try to get as many columns
    N1_tile = input_buffer_size / M_mat;
    N0_tile = input_buffer_size / (M_mat + N1_tile);

    // Assume that at least one row and output can fit
    assert(N0_tile != 0);
    EdgeBert_init(dev, plic_dev, mem);

    // Allocate memory for matrices
    token_t *left;
    token_t *right;
    token_t *output;

    left = aligned_malloc(N0_tile * M_mat);
    right = aligned_malloc(M_mat * N1_tile);
    output = aligned_malloc(N0 * N1);

    // Tranpose for easier access
    CPU_transpose(D_mat2, M_mat, N1);

    int count = 0;
    int row = 0, col = 0;
    while (row < N0) {
        // Get left matrix
        unsigned N0_mat = min(N0_tile, N0 - row);
        memcpy(left, D_mat1 + M_mat * row, N0_mat * M_mat * sizeof(token_t));
        while (col < N1) {
            // Get right matrix
            unsigned N1_mat = min(N1_tile, N1 - col);
            memcpy(right, D_mat + M_mat * col, N1_mat * M_mat * sizeof(token_t));
            CPU_transpose(right, N1_mat, M_mat);

            // Multiply
            EdgeBert_mat_mul(dev, plic_dev, N0_mat, N1_mat, M_mat, is_relu, mem, mask_mat, left, right, softmax);

            // Copy over data into relu_output
            for (int l = 0; l < N0_mat; l++) {
                for (int k = 0; k < N1_mat; k++) {
                    relu_output[(row + l) * N1 + col + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * N0_mat + k];
                }
            }
            col += N1_mat;
            count++;
        }
        row += N0_mat;
    }
}

// Softmax for attention head
static int EdgeBert_atten_softmax(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    int N0,
    int N1,
    int M_mat
) {
    printf("STARTing attention softmax in EdgeBert...\n");
    unsigned data = 0;
    int num_interrupts;

    // Calculate softmax
    // Set use_gb and reset_mode
    // Choose decoder 0
    data = 0x0;
    iowrite32(dev, 0x08, data);
    // Set SFU to true
    data = 0x1;
    iowrite32(dev, 0x50, data);
    // Set softmax for decoder 0
    data = 0x08;
    iowrite32(dev, 0x58, data);
    // Set mode to softmax
    data = 0x02;
    iowrite32(dev, 0x54, data);
    // Configure parameters
    data = 0;
    data += base_attn_span;
    data += base_gamma << 7;
    data += base_beta << 15;
    data += adpbias_attn_span << 23;
    data += adpbias_gamma << 26;
    data += adpbias_beta << 29;
    iowrite32(dev, 0x1C, data);
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    iowrite32(dev, 0x20, data);
    // Configure matrix size
    data = 0x0;
    data += N0;
    data += N1 << 10;
    data += M_mat << 20;
    iowrite32(dev, 0x10, data);
    // Set base of matrix
    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);
    // Set to decoder 0
    data = 0x0;
    iowrite32(dev, 0x08, data);
    // Start softmax
    data = 0x9;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Master input write (load from accelerator DATA scratchpad and store outside)
    // Read data to store outside
    data = 0x1;
    iowrite32(dev, 0x4C, data);
    // Start input master write
    data = 0x04;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    printf("FINISHing SoftAttenM in EdgeBert...\n");
    return num_interrupts;
}

static void EdgeBert_highway_layer() {
    // Highway exit
    // Index in to layer outputs
    token_t *input;
    memcpy(input, val_mat, N1 * sizeof(token_t));

    // Intialize linear layer and outputs
    token_t *we_mat_1;
    token_t *result_mat_1;

    N0 = he_layer1;
    M_mat = input_m;
    N1 = 1;

    we_mat_1 = aligned_malloc(N0 * M_mat);
    result_mat_1 = aligned_malloc(N0 * N1);
    memset(we_mat_1, 35, N0 * M_mat * sizeof(token_t));

    // Perform matrix multiplication
    softmax = 0;
    general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, we_mat_1, input, softmax);
    memcpy(result_mat_1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Perform classification
    token_t *we_mat_2;
    token_t *result_mat_2;

    N0 = he_layer1;
    M_mat = he_layer2;
    N1 = 1;

    we_mat_2 = aligned_malloc(N0 * M_mat);
    memset(we_mat_2, -24, N0 * M_mat * sizeof(token_t));

    // Perform matrix multiplication
    general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, we_mat_2, result_mat_1, softmax);
    memcpy(result_mat_2, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));
}

// Attention head
static void EdgeBert_attention(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int input_m,
    int input_n,
    int hidden_size,
    int he_layer1,
    int he_layer2
) {
    printf("STARTing Attention Head in EdgeBert...\n");
    int num_interrupts;
    int softmax = 0;

    // Initialize inputs and weights
    token_t *input_ids;
    token_t *we_query;
    token_t *we_key;
    token_t *we_val;
    token_t *mask_mat;
    // token_t *aux_mat;

    // Initialize IDs
    input_ids = aligned_malloc(input_m * input_n);
    we_query = aligned_malloc(input_n * hidden_size);
    we_key = aligned_malloc(input_n * hidden_size);
    we_val = aligned_malloc(input_n * hidden_size);
    mask_mat = aligned_malloc(2 * input_m * input_n);
    // aux_mat = aligned_malloc(4096);

    // Initialize weights
    // init_buf_attention(input_ids, we_mat1, we_mat2, we_mat3, mask_mat, aux_mat);

    // Fill with dummy data
    memset(input_ids, 11, input_m * input_n * sizeof(token_t));
    memset(we_query, 35, input_n * hidden_size * sizeof(token_t));
    memset(we_key, -1, input_n * hidden_size * sizeof(token_t));
    memset(we_val, -12, input_n * hidden_size * sizeof(token_t));
    // QUESTION: Mask size
    memset(mask_mat, 255, 2 * input_m * input_n * sizeof(token_t));
    // memset(aux_mat, 3, 4096 * sizeof(token_t));

    unsigned N0;
    unsigned N1;
    unsigned M_mat;
    unsigned is_relu;

    N0 = input_m;
    M_mat = input_n;
    N1 = hidden_size;
    is_relu = 0;

    token_t *query_mat;
    token_t *key_mat;
    token_t *val_mat;

    // Initialize output matrices
    query_mat = aligned_malloc(N0 * N1);
    key_mat = aligned_malloc(N0 * N1);
    value_mat = aligned_malloc(N0 * N1);

    // EdgeBert_init(dev, plic_dev, mem);
    // Mutliply IDs by query matrix
    general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids, we_query, softmax);
    // Save result
    memcpy(query_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // EdgeBert_init(dev, plic_dev, mem);
    // Mutliply IDs by key matrix
    general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids, we_key, softmax);
    // Save result
    memcpy(key_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // EdgeBert_init(dev, plic_dev, mem);
    // Mutliply IDs by value matrix
    general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids, we_val, softmax);
    // Save result
    memcpy(val_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Transpose output of key multiplication
    CPU_transpose(key_mat, N0, N1);

    // Multiply query and key outpput
    N0 = input_m;
    M_mat = input_n;
    N1 = input_m;
    token_t *query_key_mat;
    query_key_mat = aligned_malloc(N0 * N1);

    // EdgeBert_init(dev, plic_dev, mem);
    // Set softmax parameter to true
    softmax = 1;
    general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, query_mat, key_mat, softmax);

    // Free memory
    aligned_free(query_mat);
    aligned_free(key_mat);

    // Softmax and attention span configuration
    N0 = input_m;
    M_mat = input_m;
    N1 = input_m;

    // Apply softmax
    EdgeBert_atten_softmax(dev, plic_dev, N0, N1, M_mat);
    memcpy(query_key_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Multiply query and key with value matrix
    N0 = input_m;
    M_mat = input_m;
    N1 = hidden_size;

    // EdgeBert_init(dev, plic_dev, mem);
    general_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, query_key_mat, val_mat, softmax);
    memcpy(value_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Free memory
    aligned_free(input_ids);
    aligned_free(we_query);
    aligned_free(we_key);
    aligned_free(we_val);
    aligned_free(mask_mat);
    // aligned_free(aux_mat);

    printf("FINISHing Attention Head in EdgeBert...\n");
}

static token_t *EdgeBert_attention_heads(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size,
    int he_layer1,
    int he_layer2
) {
    printf("STARTing EdgeBERT %i Attention Heads Computation...\n", num_heads);
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    token_t* attention_heads;
    attention_heads = aligned_malloc(input_m * input_n);

    for (int i = 0; i < 12; i++) {
        // Run attention head
        count1 = get_counter();
        EdgeBert_attention(&dev, &plic_dev, mem);
        count2 = get_counter();
        exe_cycle = count2 - count1;
        printf("...Attention Head %d takes %"PRIu64" clock cycles...\n", i, exe_cycle);

        for (int l = 0; l < input_m; l++) {
            for (int k = 0; k < hidden_size; k++) {
                attention_heads[l * hidden_size * num_heads + i * hidden_size + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * hidden_size + k];
            }
        }

        // Keep track of number of cycles
        total_exe_cycle = total_exe_cycle + exe_cycle;
    }

    printf("FINISHing EdgeBERT 12 Attention Heads Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
    return attention_heads;
}

// Apply layer norm
static void EdgeBert_element_add_layer_norm(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    int N0,
    int N1,
    int M_mat,
    token_t *mem,
    token_t *D_mat1,
    token_t *D_mat2
) {
    unsigned data = 0;
    int num_interrupts;

    // Copy over data into the CPU
    unsigned input_rd1_base = ((unsigned) mem) + mask_buffer_size;
    unsigned input_rd2_base = ((unsigned) mem) + mask_buffer_size + input_buffer_size;
    memcpy(mem + mask_buffer_size, D_mat1, N0 * M_mat * sizeof(token_t));
    memcpy(mem + mask_buffer_size + input_buffer_size, D_mat2, M_mat * N1 * sizeof(token_t));

    // Perform element-wise addition on D_mat1 + D_mat2
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
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    iowrite32(dev, 0x20, data);

    // Load in first matrix
    data = 0x0;
    iowrite32(dev, 0x4C, data);
    // Set to decoder 0
    iowrite32(dev, 0x08, data);
    // Set act/weight data address
    data = input_rd1_base;
    iowrite32(dev, 0x30, data);
    // Start master input read
    data = 0x03;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Load in second matrix
    // Set to decoder 1
    data = 0x1;
    iowrite32(dev, 0x08, data);
    // Set read address
    data = input_rd2_base;
    iowrite32(dev, 0x30, data);
    // Start master input read
    data = 0x03;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Do the addition
    // Set matrix config
    data = 0x0;
    data += N0;
    data += N1 << 10;
    data += M_mat << 20;
    iowrite32(dev, 0x10, data);
    // Set base input
    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);
    // Start element-wise addition
    data = 0xA;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Layer norm
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
    data += base_attn_span;
    data += base_gamma << 7;
    data += base_beta << 15;
    data += adpbias_attn_span << 23;
    data += adpbias_gamma << 26;
    data += adpbias_beta << 29;
    iowrite32(dev, 0x1C, data);
    data = 0;
    data += num_vector;
    data += num_timestep << 8;
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    iowrite32(dev, 0x20, data);
    // Set base input
    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    iowrite32(dev, 0x14, data);
    // Set offset
    data = 0;
    iowrite32(dev, 0x18, data);
    // Start layer norm
    data = 0x8;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);

    // Write output to outside
    // Set up write to outside
    data = 0x1;
    iowrite32(dev, 0x4C, data);
    // Start master input write
    data = 0x04;
    iowrite32(dev, 0x04, data);
    // Wait for interrupt
    num_interrupts = wait(plic_dev, num_interrupts);
}

static void EdgeBert_processing(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    token_t *attention_out
) {
    printf("\nSTARTing 12 Attention Heads Processing...\n");
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    count1 = get_counter();
    // Multiply conatenated output (128 x 768)
    token_t* we_heads;
    we_heads = aligned_malloc(768 * 768);
    // Fill with dummy data
    memset(we_heads, -1, 768 * 768 * sizeof(token_t));

    token_t* attention_head_out;
    attention_head_out = aligned_malloc(128 * 768);
    // Fill with dummy data
    memset(attention_head_out, 100, 128 * 768 * sizeof(token_t));

    token_t *mask_mat;
    mask_mat = aligned_malloc(8192);
    // Fill with dummy data
    memset(mask_mat, 255, 8192 * sizeof(token_t));

    // Use accelerator and split input into two
    token_t *input_1;
    token_t *input_2;

    // Matrix multiplication configurations
    unsigned N0;
    unsigned N1;
    unsigned M_mat;
    unsigned is_relu;

    N0 = 64;
    M_mat = 768;
    N1 = 64;
    is_relu = 0;

    input_1 = aligned_malloc(N0 * M_mat);
    input_2 = aligned_malloc(M_mat * N1);

    EdgeBert_init(&dev, &plic_dev, mem);
    int count = 0;
    for (int i = 0; i < 2; i++) {
        memcpy(input_1, attention_heads + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));
        for (int j = 0; j < 12; j++) {
            memcpy(input_2, We_heads + j * M_mat * N1, M_mat * N1 * sizeof(token_t));

            if (count == 2) {
                EdgeBert_init(&dev, &plic_dev, mem);
                count = 0;
            }

            EdgeBert_mat_mul (&dev, &plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_1, input_2, 0);
            for (int l = 0; l < N0; l++) {
                for (int k = 0; k < N1; k++) {
                    attention_head_out[(l + i * N0) * 768 + j * N1 + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + k + l * N0];
                }
            }
            count++;
        }
    }

    aligned_free(input_1);
    aligned_free(input_2);

    N0 = 64;
    M_mat = 768;
    N1 = 64;

    token_t* attention_out;
    input_1 = aligned_malloc(N0 * M_mat);
    input_2  = aligned_malloc(M_mat * N1);

    // Add on input IDs and layer norm
    token_t* input_ids;
    input_ids = aligned_malloc(128 * 768);
    memset(input_ids, -1, 128 * 768 * sizeof(token_t));
    CPU_transpose(input_ids, 768, 128);
    attention_out = aligned_malloc(128 * 768);

    for (int i = 0; i < 2; i++) {
        // Split 128 x 768 into two 64 x 768
        memcpy(input_1, attention_head_out + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));
        memcpy(input_2, input_ids + i * N1 * M_mat, N1 * M_mat * sizeof(token_t));
        EdgeBert_init(&dev, &plic_dev, mem);
        EdgeBert_element_add_layer_norm(&dev, &plic_dev, N0, N1, M_mat, mem, input_1, input_2);
        memcpy(attention_out + i * N0 * M_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * M_mat * sizeof(token_t));
    }
    count2 = get_counter();
    printf("FINISHing 12 Attention Heads Processing...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);

    return attention_out;
}

// Feed forward
static void EdgeBert_feed_forward(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    token_t *processing_out
) {
    printf("\nSTARTing EdgeBERT Feed Forward Net Computation...\n");
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    count1 = get_counter();
    EdgeBert_init(&dev, &plic_dev, mem);

    int softmax = 0;
    token_t *mask_mat;
    mask_mat = aligned_malloc(8192);
    memset(mask_mat, 255, 8192 * sizeof(token_t));

    // Initialize weights
    token_t* we_mat1; // 768 x 3072
    token_t* we_mat2; // 3072 x 768

    we_mat1 = aligned_malloc(768 * 3072);
    we_mat2 = aligned_malloc(768 * 3072);

    // init_buf_ffn(we_mat1, we_mat2);
    // Load dummy data
    memset(we_mat1, -1, 768 * 3072 * sizeof(token_t));
    memset(we_mat2, -12, 768 * 3072 * sizeof(token_t));

    // Multiply attention output by weights ((128 x 768) x (768 x 3072) = (128 x 3072))
    unsigned N0;
    unsigned N1;
    unsigned M_mat;
    unsigned is_relu;

    N0 = 64;
    M_mat = 768;
    N1 = 64;
    is_relu = 1;

    // Split attention output into two
    token_t *input_1;     // 64 x 768
    token_t *input_2;     // 768 x 64
    token_t *relu_output; // 128 x 3072

    input_1 = aligned_malloc(N0 * M_mat);
    input_2 = aligned_malloc(M_mat * N1);
    relu_output = aligned_malloc(128 * 3072);

    EdgeBert_init(dev, plic_dev, mem);
    int count = 0;

    for (int i = 0; i < 2; i++) {
        // Load in half of attention output
        memcpy(input_1, processing_out + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));

        for (int j = 0; j < 48; j++) {
            // TODO: Need to transpose?
            memcpy(input_2, we1_mat + j * M_mat * N1, M_mat * N1 * sizeof(token_t));

            if (count == 2) {
                EdgeBert_init(dev, plic_dev, mem);
                count = 0;
            }
            EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_1, input_2, softmax);
            // memcpy(relu_output + N0 * N1 * i * j, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

            // Copy over data into relu_output
            for (int l = 0; l < N0; l++) {
                for (int k = 0; k < N1; k++) {
                    relu_output[(l + i * N0) * 3072 + j * N1 + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * N0 + k];
                }
            }
            count++;
        }
    }

    // Multiply relu_out with second weights ((8 x 16) x 3072) x (3072 x (16 x 48))
    N0 = 16;
    M_mat = 3072;
    N1 = 16;
    is_relu = 0;

    aligned_free(input_1);
    aligned_free(input_2);
    token_t *we2_output;

    input_1 = aligned_malloc(N0 * M_mat);
    input_2 = aligned_malloc(M_mat * N1);
    we2_output = aligned_malloc(128 * 768);

    // EdgeBert_init(dev, plic_dev, mem);
    count = 0;

    for (int i = 0; i < 8; i++) {
        memcpy(input_1, relu_output + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));

        for (int j = 0; j < 48; j++) {
            memcpy(input_2, we2_mat + j * M_mat * N1, M_mat * N1 * sizeof(token_t));

            if (count == 2) {
                // EdgeBert_init(dev, plic_dev, mem);
                count = 0;
            }

            EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_1, input_2, softmax);
            // memcpy(we2_output + N0 * N1 * i * j, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

            for (int l = 0; l < N0; l++) {
                for (int k = 0; k < N1; k++) {
                    we2_output[(l + i * N0) * 768 + j * N1 + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * N0 + k];
                }
            }
            count++;
        }
    }

    aligned_free(input_1);
    aligned_free(input_2);

    N0 = 64;
    M_mat = 768;
    N1 = 64;

    // Add attention output
    token_t* FFN_output;
    input_1 = aligned_malloc(N0 * M_mat);
    input_2  = aligned_malloc(M_mat * N1);
    CPU_transpose(processing_out, 768, 128);
    FFN_output = aligned_malloc(128*768);

    for (int i = 0; i < 2; i++) {
        // Add parts of attention output
        memcpy(input_1, we2_output + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));
        memcpy(input_2, processing_out + i * N1 * M_mat, N1 * M_mat * sizeof(token_t));
        EdgeBert_init(dev, plic_dev, mem);
        EdgeBert_ElementAddLayerNorm(dev, plic_dev, N0, N1, M_mat, mem, input_1, input_2);
        memcpy(FFN_output + i * N0 * M_mat, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * M_mat * sizeof(token_t));
    }

    // Free memory
    aligned_free(input_1);
    aligned_free(input_2);
    aligned_free(we2_output);
    aligned_free(mask_mat);
    aligned_free(we1_mat);
    aligned_free(we2_mat);
    aligned_free(relu_output);

    count2 = get_counter();
    printf("FINISHing EdgeBERT Feed Forward Net Computation...\n");
    printf("###(taking %"PRIu64" clock cycles)###...\n", count2 - count1);
}

static void EdgeBert_transformer(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    int num_heads,
    int input_m,
    int input_n,
    int hidden_size
) {
    printf("\n");
    printf("  #######  ######      ######       ####    #     #    #####   \n");
    printf("  #        #     #    #      #     #        #     #   #     #  \n");
    printf("  #        #     #   #        #   #         #     #   #        \n");
    printf("  #######  ######    #        #   #         #######    #####   \n");
    printf("  #        #         #        #   #         #     #         #  \n");
    printf("  #        #          #      #     #        #     #   #     #  \n");
    printf("  #######  #           ######       ####    #     #    #####   \n");
    printf("\n");

    // Attention heads
    token_t *attention_heads = EdgeBert_attention_heads(dev, plic_dev, mem, num_heads, input_m, input_n, hidden_size);

    // Processing
    token_t* attention_out = EdgeBert_processing(&dev, &plic_dev, mem, attention_heads);

    // Feed Forward Neural Net
    EdgeBert_feed_forward(&dev, &plic_dev, mem, attention_out);

    printf("\nEdgeBERT Transformer Layer DONE...\n");
    printf("Thank you!\n");

    aligned_free(attention_heads);
    aligned_free(attention_out);
}

// Driver
// Edgebert compuatation
int main(int argc, char * argv[]) {
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
    mem_size = mask_buffer_size + aux_buffer_size + 3 * input_buffer_size;
    // Allocation of the accelerator data array (mem)
    mem = aligned_malloc(mem_size);

    // Flush (customize coherence model here)
    coherence = ACC_COH_RECALL;
    coh_dev.addr = CSR_TILE_ADDR;
    iowrite32(&coh_dev, CSR_REG_OFFSET * 4, coherence);
    if (coherence != ACC_COH_RECALL) {
        esp_flush(coherence, 4);
    }

    // Run transformer on accelerator
    EdgeBert_transformer(
        dev,
        plic_dev,
        mem,
        12,
        128,
        768,
        64
    );
    aligned_free(mem);
    return 0;
}
