import torch

def main():
    # data = [[1,2,3],[4,5,6]]    
    # tensor = torch.tensor(data)
    # print(tensor)
    # data_3d = [
    #     [
    #         [1, 2, 3],
    #         [4, 5, 6]
    #     ],
    #     [
    #         [7, 8, 9],
    #         [10, 11, 12]
    #     ]
    # ]
    # tensor_3d = torch.tensor(data_3d)
    # print(tensor_3d)
    # data_4d = [
    #     [
    #         [[1, 2], [3, 4]],
    #         [[5, 6], [7, 8]]
    #     ],
    #     [
    #         [[9, 10], [11, 12]],
    #         [[13, 14], [15, 16]]
    #     ]
    # ]
    # tensor_4d = torch.tensor(data_4d)
    # print(tensor_4d)
    # shape = (2,3)
    # tensor_zeros = torch.zeros(shape)
    # tensor_ones = torch.ones(shape)
    # tensor_eye = torch.eye(3)
    # tensor_rnd = torch.rand(shape)
    # print("Zeros Tensor:\n", tensor_zeros)
    # print("Ones Tensor:\n", tensor_ones)
    # print("Identity Matrix:\n", tensor_eye)
    # print("Random Tensor:\n", tensor_rnd)   
    # data = [[1, 2, 3], [4, 5, 6]]
    # tensor_2d = torch.tensor(data)
    # tensor_2d_float = torch.tensor(tensor_2d, dtype=torch.float)
    # print(tensor_2d_float)
    # print(f"Tensor as float:\n{tensor_2d_float}")
    # tensor_rand_like = torch.rand_like(tensor_2d_float)
    # print(f"Random tensor with same shape as tensor_2d_float:\n{tensor_rand_like}")
    # data = torch.rand((2, 3))
    # print(f"Original Tensor:\n{data.shape}")
    # print(f"Original Tensor:\n{data.dtype}")
    # print(f"Original Tensor:\n{data.device}")
    x = torch.arange(12).reshape(3, 4)
    print(f"Original Tensor:\n{x}")
    x_reshaped = x.reshape(4, 3)
    print(f"Reshaped Tensor (4,3):\n{x_reshaped}")
    x_flattened = x.reshape(-1)
    print(f"Flattened Tensor:\n{x_flattened}")
if __name__ == "__main__":
    main()
