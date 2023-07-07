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
#define PLIC_IP_OFFSET 0x1000 // Interrupt Pending
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

// QUESTION: What are M and N?
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
// Transpose a matrix of chars at array with size mxn (in-place)
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
void CPU_transpose_int (int *array, int m, int n) {
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

// Softmax over array of size size
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

// Profile CPU attention head
static void CPU_EdgeBert_attention_profile() {
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

    input_ids = aligned_malloc(128 * 768 * sizeof(int));
    we_query = aligned_malloc(768 * 64 * sizeof(int));
    we_key = aligned_malloc(768 * 64 * sizeof(int));
    we_val = aligned_malloc(768 * 64 * sizeof(int));
    output1 = aligned_malloc(128 * 64 * sizeof(int));
    output2 = aligned_malloc(128 * 64 * sizeof(int));
    output3 = aligned_malloc(128 * 64 * sizeof(int));

    // Fill with dummy data
    for (int i = 0; i < 128 * 768; i++) {
        input_ids[i] = 12;
    }

    for (int i = 0; i < 768 * 64; i++) {
        we_query[i] = 24;
        we_key[i] = -5;
        we_val[i] = 126;
    }

    // Matrix multiplication configurations
    int N0;
    int N1;
    int M_mat;

    N0 = 128; M_mat = 768; N1 = 64;
    // Query multiplication
    CPU_multiply(input_ids, we_query, N0, M_mat, N1, output1);
    // Key multiplication
    CPU_multiply(input_ids, we_key, N0, M_mat, N1, output2);
    // Value multiplication
    CPU_multiply(input_ids, we_val, N0, M_mat, N1, output3);
    // Transpose key ooutput
    CPU_transpose_int(output2, N0, N1);

    // Query output (128 x 64) multiplied by transpose of key output (64 x 128)
    N0 = 128; M_mat = 64; N1 = 128;
    int *output4;
    output4 = aligned_malloc(128 * 128 * sizeof(int));
    CPU_multiply(output1, output2, N0, M_mat, N1, output4);

    // Softmax?
    // Attention Span Mask?

    // Multiply value output
    N0 = 128; M_mat = 128; N1 = 64;
    int *output5;
    output5 = aligned_malloc(128 * 64 * sizeof(int));
    CPU_multiply(output4, output3, N0, M_mat, N1, output5);

    aligned_free(input_ids);
    aligned_free(we_query);
    aligned_free(we_key);
    aligned_free(we_val);
    aligned_free(output1);
    aligned_free(output2);
    aligned_free(output3);
    aligned_free(output4);
    aligned_free(output5);
}

// Feed forward neural network after attention heads
static void CPU_EdgeBert_feed_forward() {
    int *attention_head_out;
    int *we1;
    int *we2;
    int *output1;
    int *output2;

    attention_head_out = aligned_malloc(128 * 768 * sizeof(int));
    we1 = aligned_malloc(768 * 3072 * sizeof(int));
    we2 = aligned_malloc(3072 * 768 * sizeof(int));
    output1 = aligned_malloc(128 * 3072 * sizeof(int));
    output2 = aligned_malloc(128 * 768 * sizeof(int));

    // Fill with dummy data
    for (int i = 0; i < 128 * 768; i++) {
        attention_head_out[i] = 38;
    }

    for (int i = 0; i < 3072 * 768; i++) {
        we1[i] = 24;
        we2[i] = -5;
    }

    // Matrix configuration
    int N0;
    int N1;
    int M_mat;
    N0 = 128; M_mat = 768; N1 = 3072;

    // First multiplication with attention output
    CPU_multiply(attention_head_out, we1, N0, M_mat, N1, output1);

    // Activation function?

    // Second multiplication
    N0 = 128; M_mat = 3072; N1 = 768;
    CPU_multiply(output1, we2, N0, M_mat, N1, output2);

    // Layer normalization?
}

// Profile CPU performance
static void CPU_profile() {
    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    // CPU Performance
    printf("\n\n");
    // Initialize input ids
    int *attention_heads_cpu;
    attention_heads_cpu = aligned_malloc(128 * 768 * sizeof(int));

    printf("Transformer Matmul Performance Profiling on Ariane RISC-V CPU\n");
    printf("\nSTARTing CPU 12 Attention Heads Computation...\n");

    total_exe_cycle = 0;
    for (int i = 0; i < 12; i++) {
        count1 = get_counter();
        CPU_EdgeBert_attention_profile();
        count2 = get_counter();
        exe_cycle = count2 - count1;
        printf("...Attention Head %d takes %"PRIu64" clock cycles...\n", i, exe_cycle);

        // Fill with dummy data
        for (int l = 0; l < 128; l++) {
            for (int k = 0; k < 64; k++) {
                attention_heads_cpu[l * 768 + i * 64 + k] = l * 64 + k;
            }
        }
        total_exe_cycle = total_exe_cycle + exe_cycle;
    }

    printf("FINISHing CPU 12 Attention Heads Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);
    printf("\nSTARTing CPU 12 Attention Heads Processing...\n");

    int *we_heads_cpu;
    we_heads_cpu = aligned_malloc(768 * 768 * sizeof(int));
    int *attention_head_out_cpu;
    attention_head_out_cpu = aligned_malloc(128 * 768 * sizeof(int));

    count1 = get_counter();
    for (int i = 0; i < 768 * 768; i++) {
        we_heads_cpu[i] = -1;
    }
    N0 = 128; M_mat = 768; N1 = 768;
    CPU_multiply(attention_heads_cpu, we_heads_cpu, N0, M_mat, N1, attention_head_out_cpu);
    count2 = get_counter();

    // Layer normalization?

    printf("FINISHing CPU 12 Attention Heads Processing...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);
    printf("\nSTARTing CPU Feed Forward Net Computation...\n");

    // Feed forward on CPU
    count1 = get_counter();
    CPU_EdgeBert_feed_forward();
    count2 = get_counter();

    printf("FINISHing CPU Feed Forward Net Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", count2 - count1);
    printf("\nTransformer Matmul Performance Profiling on Ariane RISC-V CPU DONE...\n");
    printf("Thank you!\n");
}


// Accelerator functions
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
    // Store base output of computations
    data += base_output;
    iowrite32(dev, 0x48, data);
    // Not using SFU
    data = 0;
    iowrite32(dev, 0x50, data);
    // Set mode to N0 for
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
    // printf("......waiting for 1st interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 1st interrupt\n");
    num_interrupts++;

    // Select decoder 1
    data = 0x1;
    iowrite32(dev, 0x08, data);

    // Start master mask read
    data = 0x01;
    iowrite32(dev,0x04, data);

    // Wait for interrupt
    // printf("......waiting for 2nd interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 2nd interrupt\n");
    num_interrupts++;

    // Set num words to read/write Aux AXI
    data = 0;
    data += (M - 1);
    iowrite32(dev, 0x44, data);

    // Set aux read base
    data = aux_rd_base;
    iowrite32(dev, 0x38, data);

    // Start aux master read
    data = 0x05;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    // printf("......waiting for the 3rd interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 3rd interrupt\n");
    num_interrupts++;

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
    printf("...STARTing Matmul in EdgeBert...\n");
    int num_interrupts = 0;

    unsigned data = 0;
    unsigned input_rd1_base = ((unsigned) mem) + mask_buffer_size;
    unsigned input_rd2_base = ((unsigned) mem) + mask_buffer_size + input_buffer_size;

    // Loads in matrices from memory
    memcpy(mem, mask_mat, mask_buffer_size * sizeof(token_t));
    memcpy(mem + mask_buffer_size, D_mat1, N0 * M_mat * sizeof(token_t));
    memcpy(mem + mask_buffer_size + input_buffer_size, D_mat2, M_mat * N1 * sizeof(token_t));
    //memcpy(mem + mask_buffer_size + 2 * input_buffer_size, aux_mat, aux_buffer_size * sizeof(token_t));

    // Set post-processing configuration
    data = 0;
    data += is_relu;
    data += is_bias << 4;
    data += weight_bias << 8;
    data += adf_accum_bias << 16;
    data += accum_right_shift << 20;
    iowrite32(dev, 0x0C, data);

    // Load in matrices to accelerator
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
    // printf("......waiting for 1st interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 1st interrupt\n");
    num_interrupts++;

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

    // Wait for interrupt
    // printf("......waiting for 2nd interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 2nd interrupt\n");
    num_interrupts++;

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
    // printf("......waiting for 3rd interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 3rd interrupt\n");
    num_interrupts++;

    // Write data outside
    if (softmax == 0) {
        // Set use_axi to 1
        data = 0x1;
        iowrite32(dev, 0x4C, data);

        // Start input master write
        data = 0x04;
        iowrite32(dev, 0x04, data);

        // Wait for interrupt
        // printf("......waiting for 4th interrupt\n");
        // iointerrupt();
        while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
        iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
        iowrite32(plic_dev, 0x2000, 0x40);
        iowrite32(plic_dev, 0x18, 0x2);
        ioread32(plic_dev, PLIC_INTACK_OFFSET);
        // printf("......receiving the 4th interrupt\n");
        num_interrupts++;
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
    unsigned N0_tile;
    unsigned N1_tile;

    // Try to get as many columns
    N1_tile = input_buffer_size / M_mat;
    N0_tile = input_buffer_size / (M_mat + N1_tile);
    // Assume that at least one row and output can fit
    assert(N0_tile != 0);

    EdgeBert_init(dev, plic_dev, mem);
    int count = 0;

    token_t *left;
    token_t *right;
    token_t *output;

    left = aligned_malloc(N0_tile * M_mat);
    right = aligned_malloc(M_mat * N1_tile);
    output = aligned_malloc(N0 * N1);

    // Tranpose for easier access
    CPU_transpose(D_mat2, M_mat, N1);

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
    data = 0;
    iowrite32(dev, 0x18, data);

    // Set to decoder 0
    data = 0x0;
    iowrite32(dev, 0x08, data);
    // Start softmax
    data = 0x9;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    // printf("...waiting for 1st interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("...receiving the 1st interrupt\n");
    num_interrupts++;

    // Read data to store outside
    data = 0x1;
    iowrite32(dev, 0x4C, data);

    // Start input master write
    data = 0x04;
    iowrite32(dev, 0x04, data);

    // Wait for interrupt
    // printf("......waiting for 4th interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 4th interrupt\n");
    num_interrupts++;

    printf("FINISHing SoftAttenM in EdgeBert...\n");
    return num_interrupts;
}

// Attention head
static void EdgeBert_attention(struct esp_device *dev, struct esp_device *plic_dev, token_t *mem) {
    printf("STARTing Attention Head in EdgeBert...\n");
    int num_interrupts;
    int softmax = 0;

    // Initialize inputs and weights
    token_t *input_ids1st;
    token_t *input_ids2nd;
    token_t *we_query;
    token_t *we_key;
    token_t *we_val;
    token_t *mask_mat;
    //token_t *aux_mat;

    // Initialize IDs
    input_ids1st = aligned_malloc(64 * 768);
    input_ids2nd = aligned_malloc(64 * 768);
    we_query = aligned_malloc(768 * 64);
    we_key = aligned_malloc(768 * 64);
    we_val = aligned_malloc(768 * 64);
    mask_mat = aligned_malloc(8192);
    // aux_mat = aligned_malloc(4096);

    // Initialize weights
    // init_buf(input_ids1st, input_ids2nd, we_mat1, we_mat2, we_mat3, mask_mat, aux_mat);

    // Fill with dummy data
    memset(input_ids1st, 11, 64 * 768 * sizeof(token_t));
    memset(input_ids2nd, 115, 64 * 768 * sizeof(token_t));
    memset(we_query, 35, 768 * 64 * sizeof(token_t));
    memset(we_key, -1, 768 * 64 * sizeof(token_t));
    memset(we_val, -12, 768 * 64 * sizeof(token_t));
    memset(mask_mat, 255, 8192 * sizeof(token_t));
    // memset(aux_mat, 3, 4096 * sizeof(token_t));

    // Set read and write addresses
    EdgeBert_init(dev, plic_dev, mem);

    unsigned N0;
    unsigned N1;
    unsigned M_mat;
    unsigned is_relu;

    N0 = 64;
    M_mat = 768;
    N1 = 64;
    is_relu = 0;

    token_t *query_mat_1;
    token_t *key_mat_1;
    token_t *vaule_mat_1;

    // Initialize output matrices
    query_mat_1 = aligned_malloc(2 * N0 * N1);
    key_mat_1 = aligned_malloc(2 * N0 * N1);
    vaule_mat_1 = aligned_malloc(2 * N0 * N1);

    // Mutliply first set of IDs by query matrix ((64 x 784) x (784 x 64) = 64 x 64)
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids1st, we_query, softmax);
    // Save result
    memcpy(query_mat_1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Multiply second set of IDs by query matrix
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids2nd, we_query, softmax);
    // Combine with last result
    memcpy(query_mat_1 + N0 * N1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // EdgeBert_init(dev, plic_dev, mem);
    // Mutliply first set of IDs by key matrix ((64 x 784) x (784 x 64) = 64 x 64)
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids1st, we_key, softmax);
    // Save result
    memcpy(key_mat_1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Multiply second set of IDs by key matrix
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids2nd, we_key, softmax);
    // Combine with last result
    memcpy(key_mat_1 + N0 * N1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // EdgeBert_init(dev, plic_dev, mem);
    // Mutliply first set of IDs by value matrix ((64 x 784) x (784 x 64) = 64 x 64)
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids1st, we_val, softmax);
    // Save result
    memcpy(vaule_mat_1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Multiply second set of IDs by value matrix
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, input_ids2nd, we_val, softmax);
    // Combine with last result
    memcpy(vaule_mat_1 + N0 * N1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Transpose output of key multiplication to 64 x 128
    CPU_transpose(key_mat_1, 2 * N0, N1);

    // Multiply query and key outpput
    N0 = 128;
    M_mat = 64;
    N1 = 128;
    token_t *query_mat_2; // the input for softmax: 128X128
    query_mat_2 = aligned_malloc(2 * N0 * N1);

    // EdgeBert_init(dev, plic_dev, mem);
    // Set softmax parameter to true
    softmax = 1;
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, query_mat_1, key_mat_1, softmax);

    // Free memory
    aligned_free(query_mat_1);
    aligned_free(key_mat_1);

    // Softmax and attention span configuration
    N0 = 128;
    M_mat = 128;
    N1 = 128;

    // Apply softmax
    EdgeBert_atten_softmax(dev, plic_dev, N0, N1, M_mat);
    memcpy(query_mat_2, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Multiply query and key with value matrix
    N0 = 128;
    M_mat = 128;
    N1 = 64;

    // EdgeBert_init(dev, plic_dev, mem);
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, query_mat_2, vaule_mat_1, softmax);
    memcpy(vaule_mat_1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Highway exit
    // Index in to layer outputs
    token_t *input;
    memcpy(input, value_mat_1, N1 * sizeof(token_t));

    // Intialize linear layer and outputs
    token_t *we_mat_1;
    token_t *result_mat_1;

    N0 = 128;
    M_mat = 128;
    N1 = 1;

    we_mat_1 = aligned_malloc(N0 * M_mat);
    result_mat_1 = aligned_malloc(N0 * N1);
    memset(we_mat_1, 35, N0 * M_mat * sizeof(token_t));

    // Perform matrix multiplication
    softmax = 0;
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, we_mat_1, vaule_mat_1, softmax);
    memcpy(result_mat_1, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Perform classification
    token_t *we_mat_2;
    token_t *result_mat_2;

    N0 = 128;
    M_mat = 10;
    N1 = 1;

    we_mat2 = aligned_malloc(N0 * M_mat);
    memset(we_query, -24, N0 * M_mat * sizeof(token_t));

    // Perform matrix multiplication
    EdgeBert_mat_mul(dev, plic_dev, N0, N1, M_mat, is_relu, mem, mask_mat, we_mat_2, result_mat_1, softmax);
    memcpy(result_mat_2, mem + mask_buffer_size + 2 * input_buffer_size + aux_buffer_size, N0 * N1 * sizeof(token_t));

    // Free memory
    aligned_free(input_ids1st);
    aligned_free(input_ids2nd);
    aligned_free(we_mat1);
    aligned_free(we_mat2);
    aligned_free(we_mat3);
    aligned_free(mask_mat);
    // aligned_free(aux_mat);

    printf("FINISHing Attention Head in EdgeBert...\n");
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

    uint64_t count1;
    uint64_t count2;
    uint64_t exe_cycle;

    // Copy over data into the CPU
    unsigned input_rd1_base = ((unsigned) mem) + mask_buffer_size;
    unsigned input_rd2_base = ((unsigned) mem) + mask_buffer_size + input_buffer_size;
    memcpy(mem + mask_buffer_size, D_mat1, N0 * M_mat * sizeof(token_t));
    memcpy(mem + mask_buffer_size + input_buffer_size, D_mat2, M_mat * N1 * sizeof(token_t));

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
    // printf("......waiting for 1st interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 1st interrupt\n");
    num_interrupts++;

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
    // printf("......waiting for 2nd interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 2nd interrupt\n");
    num_interrupts++;

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
    count1 = get_counter();
    printf("......WAIT for EADD interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    num_interrupts++;
    count2 = get_counter();
    exe_cycle = count2 - count1;
    printf("......GOT EADD interrupt, and it takes %"PRIu64" clock cycles...\n", exe_cycle);

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

    // Set lyear norm configs
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
    data = 0;
    iowrite32(dev, 0x18, data);

    // Start layer norm
    data = 0x8;
    iowrite32(dev, 0x04, data);

    count1 = get_counter();
    printf("wait for LayerNorm interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    num_interrupts++;
    count2 = get_counter();
    exe_cycle = count2 - count1;
    printf("......got the LayerNorm interrupt, and it takes %"PRIu64" clock cycles...\n", exe_cycle);

    // Set up write to outside
    data = 0x1;
    iowrite32(dev,  0x4C, data);

    // Start master input write
    data = 0x04;
    iowrite32(dev,  0x04, data);

    // Wait for interrupt
    // printf("......waiting for 4th interrupt\n");
    // iointerrupt();
    while((ioread32(plic_dev, PLIC_IP_OFFSET) & 0x40) == 0);
    iowrite32(plic_dev, PLIC_INTACK_OFFSET, EDGEBERT_IRQ + 1);
    iowrite32(plic_dev, 0x2000, 0x40);
    iowrite32(plic_dev, 0x18, 0x2);
    ioread32(plic_dev, PLIC_INTACK_OFFSET);
    // printf("......receiving the 4th interrupt\n");
    num_interrupts++;
}

// Feed forward
static void EdgeBert_feed_forward(
    struct esp_device *dev,
    struct esp_device *plic_dev,
    token_t *mem,
    token_t *attention_out
) {
    int softmax = 0;
    printf("STARTing EdgeBERT Feed Forward Computation...\n");
    token_t *mask_mat;
    mask_mat = aligned_malloc(8192);
    // memset(mask_mat, 255, 8192 * sizeof(token_t));

    // Initialize weights
    token_t* we1_mat; // 768 x 3072
    token_t* we2_mat; // 3072 x 768

    we1_mat = aligned_malloc(768 * 3072);
    we2_mat = aligned_malloc(768 * 3072);

    // Load dummy data
    memset(we1_mat, -1, 768 * 3072 * sizeof(token_t));
    memset(we2_mat, -12, 768 * 3072 * sizeof(token_t));

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
        memcpy(input_1, attention_out + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));

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
    CPU_transpose(attention_out, 768, 128);
    FFN_output = aligned_malloc(128*768);

    for (int i = 0; i < 2; i++) {
        // Add parts of attention output
        memcpy(input_1, we2_output + i * N0 * M_mat, N0 * M_mat * sizeof(token_t));
        memcpy(input_2, attention_out + i * N1 * M_mat, N1 * M_mat * sizeof(token_t));
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
    printf("FINISHing EdgeBERT Feed Forward Computation...\n");
}

// Input and expected output initialization
static void init_buf(token_t *input_ids1st, token_t *input_ids2nd, token_t *we_mat1,token_t *we_mat2, token_t *we_mat3, token_t *mask_mat, token_t *aux_mat) {
    // #include "input_ids1st.h" // 128*768 -> 64*768
    // #include "input_ids2nd.h" // 128*768 -> 64*768
    // #include "we_mat1.h" // 768*64
    // #include "we_mat2.h" // 768*64
    // #include "we_mat3.h" // 768*64
    // #include "mask_mat.h" // 8192 chars
    // #include "aux_mat.h" // 4096 chars
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

    int num_interrupts;

    // Start Epochs
    printf("\n");
    printf("  #######  ######      ######       ####    #     #    #####   \n");
    printf("  #        #     #    #      #     #        #     #   #     #  \n");
    printf("  #        #     #   #        #   #         #     #   #        \n");
    printf("  #######  ######    #        #   #         #######    #####   \n");
    printf("  #        #         #        #   #         #     #         #  \n");
    printf("  #        #          #      #     #        #     #   #     #  \n");
    printf("  #######  #           ######       ####    #     #    #####   \n");
    printf("\n");
    printf("STARTing EdgeBERT 12 Attention Heads Computation...\n");

    // Transformer
    token_t* attention_heads;
    attention_heads = aligned_malloc(128 * 768);

    uint64_t total_exe_cycle = 0;
    uint64_t count1;
    uint64_t count2;
    double exe_time;
    uint64_t exe_cycle;

    for (int i = 0; i < 12; i++) {
        // Run attention head
        count1 = get_counter();
        EdgeBert_attention(&dev, &plic_dev, mem);
        count2 = get_counter();
        exe_time = (count2 - count1) * 50;
        exe_cycle = (count2 - count1);
        printf("...Attention Head %d takes %u ns seconds...\n", i, exe_time);
        printf("...Attention Head %d takes %"PRIu64" clock cycles...\n", i, exe_cycle);

        for (int l = 0; l < 128; l++) {
            for (int k = 0; k < 64; k++) {
                // TODO: Look at mem memory location!
                attention_heads[l * 768 + i * 64 + k] = mem[mask_buffer_size + 2 * input_buffer_size + aux_buffer_size + l * 64 + k];
            }
        }

        // Keep track of number of cycles
        total_exe_cycle = total_exe_cycle + exe_cycle;

        // TODO: Add high way exit
    }
    printf("FINISHing EdgeBERT 12 Attention Heads Computation...\n");
    printf("###(%"PRIu64" clock cycles)###\n", total_exe_cycle);

    // Processing
    printf("\nSTARTing 12 Attention Heads Processing...\n");
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

    // Feed Forward Neural Net
    printf("\nSTARTing EdgeBERT Feed Forward Net Computation...\n");
    EdgeBert_init(&dev, &plic_dev, mem);

    count1 = get_counter();
    EdgeBert_feed_forward (&dev, &plic_dev, mem, attention_out);
    count2 = get_counter();
    printf("FINISHing EdgeBERT Feed Forward Net Computation...\n");
    printf("###(taking %"PRIu64" clock cycles)###...\n", count2 - count1);
    printf("\nEdgeBERT Transformer Layer DONE...\n");
    printf("Thank you!\n");

    aligned_free(attention_out);
    aligned_free(mem);
    return 0;
}
