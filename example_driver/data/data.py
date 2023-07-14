matrix = []
row = 768
col = 768
c = -1

def write_data(arr, name, file_name, m, n):
    with open(file_name, 'w') as f:
        for i in range(m):
            for j in range(n):
                f.write("{0}[{1}] = {2};\n".format(name, i * n + j, arr[i][j]))

if __name__ == "__main__":
    for i in range(row):
        tmp = []
        for j in range(col):
            tmp.append(c)
        matrix.append(tmp)
    write_data(matrix, "we_mat1", "we_mat1.h", row, col)
