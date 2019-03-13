from sklearn import svm
import pandas as pd
import numpy as np
import os.path as osp
from easydict import EasyDict as edict
import time

# config
cfg = edict()
cfg.data_dir = 'data'

# load train-test data
train_data = pd.read_csv(osp.join(cfg.data_dir, 'train.csv'))
test_data = pd.read_csv(osp.join(cfg.data_dir, 'test.csv'))

# process train data
train_data = train_data.values
train_label = train_data[:, 0]
train_value = train_data[:, 1:] / 255.0
test_data = test_data.values / 255.0

# train svm
model = svm.NuSVC(max_iter=100)
start_time = time.time()
model.fit(train_value, train_label)
print('Training Done, time cost {}'.format(time.time() - start_time))

# inference
start_time = time.time()
pred_label = model.predict(test_data)
print('Inference Done, time cost {}'.format(time.time() - start_time))

# write result
result = pd.DataFrame({
    'ImageId': np.arange(1, test_data.shape[0] + 1, 1),
    'Label': pred_label
})
result.to_csv('result.csv', index=False)


