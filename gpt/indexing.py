import torch

def main():
    x = torch.arange(12).reshape(3,4)
    print(x)
    col_2 = x[:,2]
    print(col_2)
    row_1_col_2_3 = x[1, 2:4]
    print(row_1_col_2_3)
if __name__ == "__main__":
    main()