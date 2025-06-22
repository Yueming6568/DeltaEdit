import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data_name = 'celeba' # 'celeba' or 'cocoval'

#for img/text
cspace_img = np.load(f'./{data_name}/cspace_{data_name}_i.npy')
cspace_text = np.load(f'./{data_name}/cspace_{data_name}_t.npy')

#for deltaimg
cspace_deltaimg = np.load(f'./{data_name}/cspace_{data_name}_deltai.npy')
cspace_deltatext = np.load(f'./{data_name}/cspace_{data_name}_deltat.npy')

num=1000

data_ori = np.concatenate([cspace_img[:num], cspace_text[:num]], axis=0)
data_delta = np.concatenate([cspace_deltaimg[:num], cspace_deltatext[:num]], axis=0)

tsne = TSNE(n_components=2, init='pca')

result_ori = tsne.fit_transform(data_ori)
result_delta = tsne.fit_transform(data_delta)

for i in range(result_ori.shape[0]):
    x_min, x_max = np.min(result_ori, 0), np.max(result_ori, 0)
    data = (result_ori - x_min) / (x_max - x_min)
    if i < result_ori.shape[0]//2:
        s0 = plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(0/4), s=12, marker='o')
    elif i < result_ori.shape[0]:
        s1 = plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(1/4), s=12, marker='o')
    
plt.legend((s0, s1), ('CLIP Image Space', 'CLIP Text Space'), fontsize=10)
plt.xticks()
plt.yticks()
plt.title('t-SNE Results')
plt.tight_layout()
plt.savefig(f'tSNE-{data_name}-{num}_ori.png')

plt.close()

for i in range(result_delta.shape[0]):
    x_min, x_max = np.min(result_delta, 0), np.max(result_delta, 0)
    data = (result_delta - x_min) / (x_max - x_min)
    if i < result_delta.shape[0]//2:
        s0 = plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(2/4), s=12, marker='o')
    elif i < result_delta.shape[0]:
        s1 = plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(3/4), s=12, marker='o')
    
plt.legend((s0, s1), ('CLIP Delta Image Space', 'CLIP Delta Text Space'), fontsize=10)
plt.xticks()
plt.yticks()
plt.title('t-SNE Results')
plt.tight_layout()
plt.savefig(f'tSNE-{data_name}-{num}_delta.png')