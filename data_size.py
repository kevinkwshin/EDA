import pandas as pd
import matplotlib.pyplot as plt

import os

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')


#the func is from https://www.kaggle.com/toshik/image-size-and-rate-of-new-whale
def get_size_list(targets, dir_target):
    result = list()
    for target in tqdm(targets):
        img = np.array(Image.open(os.path.join(dir_target, target+'.png')))
        result.append(img.shape)
    return result

# the func is from https://www.kaggle.com/kaerunantoka/extract-image-features
def get_size(file_name_list, dir_target):
    result = list()
    #filename = images_path + filename
    for file_name in tqdm(file_name_list):
        st = os.stat(f'{dir_target}/{file_name}.png')
        result.append(st.st_size)
    return result
    
train['image_shape'] = get_size_list(train.id_code.tolist(),
                                     dir_target='../input/aptos2019-blindness-detection/train_images')
test['image_shape'] = get_size_list(test.id_code.tolist(),
                                    dir_target='../input/aptos2019-blindness-detection/test_images')
train['image_size'] = get_size(train.id_code.tolist(),
                               dir_target='../input/aptos2019-blindness-detection/train_images')
test['image_size'] = get_size(test.id_code.tolist(),
                              dir_target='../input/aptos2019-blindness-detection/test_images')
                              
for df in [train, test]:
    df['height'] = df['image_shape'].apply(lambda x:x[0])
    df['width'] = df['image_shape'].apply(lambda x:x[1])
    df['width_height_ratio'] = df['height'] / df['width']
    df['width_height_added'] = df['height'] + df['width']
    
train.head()
train.describe()
test.head()
test.describe()

fig = plt.figure(figsize=(16,10))
plt.subplot(241)
plt.hist(train['width'])
plt.title("train width")
plt.xlim(200, 4500)

sns.heatmap(train.corr(), cmap=plt.cm.Blues, annot=True);
