import pandas as pd
import scipy.sparse as sp
import numpy as np

# df = pd.read_csv('D:\E_study\大学\大三下\\research\文献\ManyDG\data\drugrec\output\diag_adj.csv')
# print(df.shape)
# s = sp.csr_matrix(df.to_numpy())
# s = s.tocoo()
# s = s.tocsr()

# sp.save_npz('spg', s)

import numpy as np
import pandas as pd

# 读取 .npz 文件
data = np.load("feature.npz")

# 遍历 npz 文件中的所有数组并保存为 CSV
for key in data.files:
    df = pd.DataFrame(data[key])  # 转换为 DataFrame
    df.to_csv(f"{key}.csv", index=False)  # 保存为 CSV
    print(f"Saved {key}.csv")