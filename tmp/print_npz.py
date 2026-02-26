import numpy as np
import sys 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_npz.py <file.npz>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = np.load(file_path, allow_pickle=True)
    
    for key in data.files:
        print(f"{key}")
        print(data[key].shape)