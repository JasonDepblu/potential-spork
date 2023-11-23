
# 导入相关模块及数据


pip install shap

pip install transformers

pip install tqdm
#%%
# 导入模块
import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import copy
import shap
import scipy
from collections import UserList
from tabulate import tabulate
from tensorflow.keras.callbacks import ModelCheckpoint
import re
from transformers import BertTokenizer, TFBertModel
from transformers import AdamW
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

#%% # Load the dataset
file_path_car = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/car_dict_e8_emb.csv'
car_data = pd.read_csv(file_path_car)

file_path_user = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/dsc_up_e8.csv'
user_data = pd.read_csv(file_path_user)

# Display the first few rows of the dataframe
car_data.head()
user_data.head()

"""# `预处理data`"""

# Remove columns with only one unique value
columns_to_drop = car_data.columns[car_data.nunique() <= 1]
car_data_reduced = car_data.drop(columns=columns_to_drop)

car_data_reduced.head(10)

# Define the new function to handle multiple ● symbols
def replace_with_detailed_feature_description(cell, feature_name):
    if pd.isna(cell):
        return cell  # Keep NaN as is
    elif isinstance(cell, str):
        # Replace specific symbols with words
        cell = cell.replace('●', '有').replace('○', '选配').replace('-', '不具有').replace('NULL', '未知')
        # Find all occurrences of "有" which indicates a feature is present
        features = cell.split()
        feature_descriptions = []
        for feature in features:
            if feature.startswith('有'):
                feature_description = feature[1:]  # Remove the '有' prefix
                feature_descriptions.append(feature_description)
        if feature_descriptions:
            return f"具备{feature_name}功能，" + "、".join(feature_descriptions)
        else:
            # If there are no '有', but the cell is not empty or NaN, it means
            # it's either '选配', '不具有' or '未知'
            return cell + feature_name + "功能"
    else:
        return cell  # Keep the original value if it doesn't match the above

# Apply the new function to each non-numeric cell
for column in car_data_reduced.columns:
    if car_data_reduced[column].dtype == 'object':  # Apply only to non-numeric columns
        feature_name = column.strip()  # Clean up the column name if necessary
        car_data_reduced[column] = car_data_reduced[column].apply(lambda cell:
                               replace_with_detailed_feature_description(cell, feature_name))

car_data_reduced.head(10)

# Define the path for the cleaned CSV file
cleaned_file_path = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/cleaned_car_data.csv'

# Save the cleaned dataframe to a CSV file
car_data_reduced.to_csv(cleaned_file_path, index=False)  # Set index to False to avoid saving the index



dsc_up = user_data.values
car_dict = car_data_reduced.values

print(dsc_up.shape)
print(car_dict.shape)

# column of last feature from 0 and +1
col_last_feature_car = 238
# 80
col_last_num_feature_car = 38

col_last_feature_user = 9

col_last_num_feature_user = 5

# 提取特征列表
dsc_up_features = user_data.columns.tolist()
car_dict_features = car_data_reduced.columns.tolist()

dsc_up_features = user_data.columns[1:col_last_feature_user].tolist() # 提取第1至28个特征
car_dict_features = car_data_reduced.columns[1:col_last_feature_car].tolist() # 提取第1至155个特征
car_dict_num_features = car_data_reduced.columns[1:col_last_num_feature_car].tolist() # 提取第151至155个特征
dsc_up_num_features = user_data.columns[1:col_last_num_feature_user].tolist()
#

print(dsc_up_features)
print(car_dict_features)
print(dsc_up_features[-1])
print(car_dict_features[-1])
print(car_dict_num_features[-1])
print(dsc_up_num_features[-1])

num_cars = len(car_dict)
print(f'The No. of cars in fundamental model dataset : {num_cars}')

# balance dsc_up samples
dsc_up_bal = np.empty(shape=(1, dsc_up.shape[1]))

for i in range(num_cars):
  if len(dsc_up[user_data.no_in_car_dict == i]) > 300:
    ran_row_bal = np.random.choice(np.arange(len(dsc_up[user_data.no_in_car_dict == i])), size=300, replace=False)
    dsc_up_bali = dsc_up[user_data.no_in_car_dict == i][ran_row_bal]
  else:
    dsc_up_bali = dsc_up[user_data.no_in_car_dict == i]
  dsc_up_bal = np.concatenate((dsc_up_bal, dsc_up_bali), axis=0)
dsc_up = dsc_up_bal[1:]
print(dsc_up.shape)

# load the label
print(f'The shape of cars dict of the fundamental model dataset : {car_dict.shape}')
print(f'The shape of users list of the fundamental model dataset : {dsc_up.shape}')

ys_up = dsc_up[:, 12]
print(f'The label of which user choosed which car in fundamental dataset : {ys_up}')

# transfer the label of fundamental dataset to onehot_code
ys = copy.copy(ys_up.reshape(len(ys_up), 1))
ys -= 1
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded = onehot_encoder.fit_transform(ys)
ys = onehot_encoded
print(f'The shape of the label of fundamental dataset to onehot_code : {ys.shape}')

car_unscaled = car_dict[:, 1:col_last_feature_car]
user_unscaled = dsc_up[:, 1:col_last_feature_user]


# generate the gaussian random no. to budget feature
mu_budget = 0.5
sigma_budget = 2

for i in range(len(user_unscaled)):
    user_unscaled[i, 4] += random.gauss(mu_budget, sigma_budget)

car = car_unscaled.copy()
user = user_unscaled.copy()

# car_num_unscaled = car_unscaled[:, 0:col_last_num_feature_car-1]
# user_num_unscaled = user_unscaled[:, (col_last_num_feature_user-1):(col_last_feature_user-1)]

# car_nnum_unscaled = car_unscaled[:, col_last_num_feature_car-1:col_last_feature_car-1]
# user_nnum_unscaled = user_unscaled[:, 0:col_last_num_feature_user-1]

# print(car_nnum_unscaled.shape)
# print(user_nnum_unscaled.shape)

# scaling the data
scalerCar = StandardScaler()
scalerCar.fit(car_unscaled[:, 0:col_last_num_feature_car-1])
car_nume = scalerCar.transform(car_unscaled[:, 0:col_last_num_feature_car-1])
car[:, 0:col_last_num_feature_car-1] = car_nume

scalerUser = StandardScaler()
scalerUser.fit(user_unscaled[:, (col_last_num_feature_user-1):(col_last_feature_user-1)])
user_nume = scalerUser.transform(user_unscaled[:, (col_last_num_feature_user-1):(col_last_feature_user-1)])
user[:, col_last_num_feature_user-1:col_last_feature_user-1] = user_nume

print(np.allclose(car_unscaled[:, 0:col_last_num_feature_car-1].astype(float), scalerCar.inverse_transform(car_nume)))
print(np.allclose(user_unscaled[:, (col_last_num_feature_user-1):(col_last_feature_user-1)].astype(float), scalerUser.inverse_transform(user_nume)))


#%%
# 样本匹配
# func for generating num_items users of different cars
def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict matrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (1, num_items))
    return user_vecs

def gen_car_vecs(car_vec, num_users):
    car_vecs = np.tile(car_vec, (num_users, 1))
    return car_vecs

# generate num_items users for different cars
user_vecs = gen_user_vecs(user, len(car_dict))
user_vecs = user_vecs.reshape(-1, user.shape[1])
car_vecs = gen_car_vecs(car, len(dsc_up))
ys = ys.reshape(-1, 1)
print(f'The shape of user matrix in fundamental dataset : {user_vecs.shape}')
print(f'The shape of car matrix in fundamental dataset : {car_vecs.shape}')
print(f'The shape of label matrix in fundamental dataset : {ys.shape}')

print(ys[ys == 0].shape)

randnum = random.randint(0, 100)
print(randnum)
random.seed(randnum)

# case 3
# Random choose the users which label is 0 in fundamental dataset

user_vecs_one = user_vecs[(ys == 1)[:, 0]]
user_vecs_zero = user_vecs[(ys == 0)[:, 0]]
num_users = user_vecs_one.shape[0]
num_zeros = user_vecs_zero.shape[0]
random.seed(randnum)

sample_row = np.random.choice(np.arange(num_zeros), size=np.ceil(num_users * 1.2).astype(int), replace=False)
user_vecs_zero = user_vecs_zero[sample_row]
user_vecs = np.concatenate((user_vecs_zero, user_vecs_one), axis=0)

# random.seed(44)
shuffle_row = np.random.choice(np.arange(0,len(user_vecs)), len(user_vecs), replace=False)
user_vecs = user_vecs[shuffle_row]
print(f'The shape of the users data in fundatmental dataset after balancing : {user_vecs.shape}')

# case3
# Random choose the cars which label is 0 in fundamental dataset

car_vecs_one = car_vecs[(ys == 1)[:, 0]]
car_vecs_zero = car_vecs[(ys == 0)[:, 0]][sample_row]
car_vecs = np.concatenate((car_vecs_zero, car_vecs_one), axis=0)

car_vecs = car_vecs[shuffle_row]
print(f'The shape of the cars data in fundatmental dataset after balancing : {car_vecs.shape}')

# case 3
# Random choose the label which is 0 in fundamental dataset
ys_one = ys[(ys == 1)[:, 0]]
ys_zero = ys[(ys == 0)[:, 0]][sample_row]
ys = np.concatenate((ys_zero, ys_one), axis=0)

ys = ys[shuffle_row]
print(f'The shape of label data in fundatmental dataset after balancing : {ys.shape}')

user_vecs_nnum = user_vecs[:, 0:col_last_num_feature_user-1]
user_vecs_num = user_vecs[:, (col_last_num_feature_user-1):(col_last_feature_user-1)]

car_vecs_nnum = car_vecs[:, col_last_num_feature_car-1:col_last_feature_car-1]
car_vecs_num = car_vecs[:, 0:col_last_num_feature_car-1]

print(user_vecs_nnum.shape)
print(user_vecs_num.shape)
print(car_vecs_nnum.shape)
print(car_vecs_num.shape)


#%%
# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model =  TFBertModel.from_pretrained('bert-base-chinese')

#%%
# preprocessing the text
def text_clean(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Convert non-string text to a string (if it's NaN, it becomes the string 'nan')
    text = str(text) if text is not None else ""

    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', '[SEP]', text).strip()
    # text =  "[CLS] " + text + "[SEP]"

    return text


#%%
def text_join(ar_features):
    features = []
    for r in range(ar_features.shape[0]):
        feature_row = []
        for c in range(ar_features.shape[1]):
            # Convert the element to string and append to the feature_row list
            feature_row.append(str(ar_features[r, c]))
        # Join the string representations of the features using a separator
        features.append(' '.join(feature_row))

    # Convert the list of strings to a NumPy array
    features = np.array(features, dtype=object)
    return features

#%%
# def batch_generator(data, batch_size):
#     """Yield successive n-sized chunks from data."""
#     for i in range(0, len(data), batch_size):
#         yield data[i:i + batch_size]

# def embeddings_for_features(features, max_length_comments, stride, batch_size):
#     features_emb_tensor = []

#     # Iterate over each batch of features
#     for batch in tqdm(batch_generator(features, batch_size), desc="Processing users"):
#         batch_input_ids = []
#         batch_attention_masks = []

#         # Process each user's features
#         for user_features in batch:
#             user_input_ids = []
#             user_attention_masks = []

#             # Process each feature text for the user
#             for feature_text in user_features:
#                 clean_text = text_clean(feature_text)

#                 # Tokenize and encode the text
#                 tokenized_inputs = tokenizer(
#                     text=clean_text,
#                     add_special_tokens=True,
#                     max_length=max_length_comments,
#                     stride=stride,
#                     return_tensors="tf",
#                     truncation=True
#                 )

#                 # Handle all segments from tokenization with stride
#                 for i in range(tokenized_inputs["input_ids"].shape[0]):
#                     user_input_ids.append(tokenized_inputs["input_ids"][i])
#                     user_attention_masks.append(tokenized_inputs["attention_mask"][i])

#             # Stack the tensors for all segments of all features of the user
#             user_input_ids = tf.stack(user_input_ids, axis=0)
#             user_attention_masks = tf.stack(user_attention_masks, axis=0)

#              # Run the model on the stacked tensors
#             outputs = model(user_input_ids, attention_mask=user_attention_masks)

#             # Aggregate the embeddings from the '[CLS]' tokens of each segment
#             cls_embeddings = outputs.last_hidden_state[:, 0, :]
#             aggregated_embedding = tf.reduce_mean(cls_embeddings, axis=0)

#             # Append the aggregated embeddings for the user to the batch list
#             features_emb_tensor.append(aggregated_embedding.numpy())

#     # Convert the list of embeddings to a tensor
#     features_emb_tensor = tf.convert_to_tensor(features_emb_tensor, dtype=tf.float32)

#     return features_emb_tensor
#%%
user_nnum = text_join(user_vecs_nnum)
print(f'joined features of users shape is {user_nnum.shape}')
#%%
encoded_comment = [tokenizer(sent, add_special_tokens=True) for sent in user_nnum]
max_length_comments = max([len(sent['input_ids']) for sent  in encoded_comment])
print('Max length: ', max_length_comments)

durations = []
for sent in encoded_comment:
    durations.append(len(sent['input_ids']))
plt.figure(figsize=(30, 5))
nums, bins, patches = plt.hist(durations, bins=20, edgecolor='k')
plt.xticks(bins, bins)
for num, bin in zip(nums, bins):
    plt.annotate(num, xy=(bin, num), xytext=(bin + 1.5, num + 0.5))
plt.show()

max_length_comments = 2048

#%%
car_nnum = text_join(car_vecs_nnum)
print(f'joined features of cars shape is {user_nnum.shape}')
#%%
encoded_comment = [tokenizer(sent, add_special_tokens=True) for sent in car_nnum]
max_length = max([len(sent['input_ids']) for sent  in encoded_comment])
print('Max length: ', max_length_comments)

durations = []
for sent in encoded_comment:
    durations.append(len(sent['input_ids']))
plt.figure(figsize=(30, 5))
nums, bins, patches = plt.hist(durations, bins=20, edgecolor='k')
plt.xticks(bins, bins)
for num, bin in zip(nums, bins):
    plt.annotate(num, xy=(bin, num), xytext=(bin + 1.5, num + 0.5))
plt.show()
#%%
max_length = 2700


#%%
## test join the features as inputs(features)
def embeddings_for_features(features, max_length_comments, batch_size_user=4):
    features_emb_list = []
    # Iterate over each user's features
    for batch_start_index in tqdm(range(0, len(features), batch_size_user), desc="Processing users"):
        batch_features = features[batch_start_index:batch_start_index+batch_size_user]
        batch_input_ids = []
        batch_attention_masks = []
        for feature_text in batch_features:
            clean_text = text_clean(feature_text if feature_text else "")
            # Tokenize and encode the text
            tokenized_inputs = tokenizer(
                text=clean_text,
                add_special_tokens=True,
                max_length=max_length_comments,
                padding='max_length',
                return_tensors="tf",
                truncation=True
            )
            batch_input_ids.append(tokenized_inputs["input_ids"][0])
            batch_attention_masks.append(tokenized_inputs["attention_mask"][0])
        batch_input_ids = tf.stack(batch_input_ids, axis=0)
        batch_attention_masks = tf.stack(batch_attention_masks, axis=0)
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        cls_embeddings = outputs.last_hidden_state.numpy()
        # Store the results in the list
        for user_features_embeddings in cls_embeddings:
            features_emb_list.append(user_features_embeddings)
            # Convert the list of embeddings to a tensor
    # features_emb_tensor = tf.convert_to_tensor(features_emb_list, dtype=tf.float32)
    return features_emb_list

#%%
# test for the embeddings function which has overflow
# def embeddings_for_features(features, max_length_comments, stride=512, batch_size_user=4):
#     features_emb_list = []
#     # Iterate over each user's features
#     for batch_start_index in tqdm(range(0, len(features), batch_size_user), desc="Processing users"):
#         batch_features = features[batch_start_index:batch_start_index+batch_size_user]
#         batch_input_ids = []
#         batch_attention_masks = []
#         for feature_text in batch_features:
#             # input_ids = []
#             # attention_mask = []
#             clean_text = text_clean(feature_text if feature_text else "")
#             # Tokenize and encode the text
#             tokenized_inputs = tokenizer(
#                 text=clean_text,
#                 add_special_tokens=True,
#                 max_length=max_length_comments,
#                 stride=stride,
#                 return_tensors="tf",
#                 truncation=True,
#                 return_overflowing_tokens=True
#             )
#             # 处理溢出的tokens
#             if tokenized_inputs["overflowing_tokens"] is not None:
#                 sep_token_tensor = tf.constant([tokenizer.sep_token_id], dtype=tf.int32)
#                 sep_token_tensor = tf.reshape(sep_token_tensor, [1, 1])
#                 sec_ids = tf.concat([tf.cast(tokenized_inputs["input_ids"], tf.int32),
#                                      tf.cast(tokenized_inputs["overflowing_tokens"], tf.int32)], axis=1)
#                 sec_ids = tf.concat([sec_ids, sep_token_tensor], axis=1)
#                 sec_att_mask = tf.ones_like(sec_ids)
#             else:
#                 sec_ids = tokenized_inputs["input_ids"]
#                 sec_att_mask = tokenized_inputs["attention_mask"]
#             batch_input_ids.append(sec_ids)
#             batch_attention_masks.append(sec_att_mask)
#         # Convert each tensor to numpy and store in a new list
#         numpy_arrays_ids = [tensor.numpy() for tensor in batch_input_ids]
#         numpy_arrays_masks = [tensor.numpy() for tensor in batch_attention_masks]
#         # Create a RaggedTensor from the list of numpy arrays
#         ragged_input_ids = tf.ragged.constant(numpy_arrays_ids)
#         ragged_attention_masks = tf.ragged.constant(numpy_arrays_masks)
#         max_length = ragged_input_ids.bounding_shape()[2]
#         # Pad the ragged tensors to the maximum length.
#         padded_input_ids = ragged_input_ids.to_tensor(shape=[None, None, max_length], default_value=tokenizer.pad_token_id)[:, 0, :]
#         padded_attention_masks = ragged_attention_masks.to_tensor(shape=[None, None, max_length], default_value=0)[:, 0, :]
#         # Run the model on the batch
#         outputs = model(padded_input_ids, attention_mask=padded_attention_masks)
#         cls_embeddings = outputs.last_hidden_state
#         # Store the results in the list
#         for i, user_features_embeddings in enumerate(cls_embeddings):
#             features_emb_list.append(user_features_embeddings)
#             # Convert the list of embeddings to a tensor
#     features_emb_tensor = tf.convert_to_tensor(features_emb_list, dtype=tf.float32)
#     return features_emb_tensor


#%%
# test
features_emb_tensor = embeddings_for_features(user_nnum, max_length_comments=2048,
                                                                    stride=512,
                                              batch_size_user=16)
#%%
print(features_emb_tensor)
# print(attention_mask_tensor)


#%%
# def embeddings_for_features(features, max_length_comments, stride):
#     # Assume embedding_dim is the dimensionality of your BERT model's embeddings
#     embedding_dim = 768
#     features_emb = np.zeros((features.shape[0], features.shape[1], embedding_dim), dtype=np.float32)
#
#     # Iterate over each user's features
#     for user_index in tqdm(range(len(features)), desc="Processing users"):
#         for feature_index in range(features.shape[1]):
#             feature_text = features[user_index][feature_index]
#             feature_text = feature_text if feature_text else ""
#             # The text_clean function needs to be defined elsewhere
#             clean_text = text_clean(feature_text)
#             # Tokenize and encode the text
#             tokenized_inputs = tokenizer(
#                 text=clean_text,
#                 add_special_tokens=True,
#                 max_length=max_length_comments,
#                 stride=stride,
#                 return_tensors="tf",
#                 truncation=True,
#                 return_overflowing_tokens = True
#             )
#
#             # Model inference using batch processing
#             outputs = model(input_ids=tokenized_inputs["input_ids"], attention_mask=tokenized_inputs["attention_mask"])
#
#             # Instead of stacking and reducing, directly take the mean of the '[CLS]' embeddings
#             # if there are multiple segments due to overflow
#             cls_embeddings = outputs.last_hidden_state[:, 0, :]
#             feature_embedding = tf.reduce_mean(cls_embeddings, axis=0).numpy()  # Average over segments if any
#
#             features_emb[user_index, feature_index, :] = feature_embedding
#
#     features_emb_tensor = tf.convert_to_tensor(features_emb, dtype=tf.float32)
#
#     return features_emb_tensor
#%%
def embeddings_for_oth_fea(features, max_length, batch_size):
    embedding_dim = 768
    features_emb_list = []
    accumulated_tensor = None
    # Tokenizer and model should be defined outside this function
    # Ensure tokenizer and model are using the correct device (CPU/GPU)
    for user_index in tqdm(range(0, len(features), batch_size), desc="Processing users"):
        batch_features = features[user_index: user_index + batch_size]
        # Tokenize all texts in the batch
        batch_input_ids = []
        batch_attention_masks = []
        for feature_texts in batch_features:
            for feature_text in feature_texts:
                clean_text = text_clean(feature_text) if feature_text else ""
                tokenized_inputs = tokenizer(
                    text=clean_text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="tf"
                )
                batch_input_ids.append(tokenized_inputs["input_ids"][0])
                batch_attention_masks.append(tokenized_inputs["attention_mask"][0])

        # Convert lists to tensors and ensure they are 2D
        batch_input_ids = tf.stack(batch_input_ids, axis=0)
        batch_attention_masks = tf.stack(batch_attention_masks, axis=0)
        # Run the model on the batch
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        cls_embeddings = outputs.last_hidden_state

        ten_a = tf.reshape(cls_embeddings,(batch_size, max_length, -1, embedding_dim))
        # ten_a = tf.reduce_mean(ten_a, axis=2)

        if accumulated_tensor is None:
            accumulated_tensor = ten_a
        else:
            accumulated_tensor = tf.concat([accumulated_tensor, ten_a], axis=0)
        # features_emb_tensor = tf.concat(features_emb_tensor, cls_embeddings)
        # Store the results in the list
    #     for user_features_embeddings in cls_embeddings:
    #         features_emb_list.append(user_features_embeddings)
    #
    # # Convert the list of embeddings to a tensor
    # features_emb_tensor = tf.convert_to_tensor(features_emb_list, dtype=tf.float32)
    # features_emb_tensor = tf.reshape(features_emb_tensor, (len(features), max_length, -1, embedding_dim))
    return accumulated_tensor

#%%
def batch_convert_to_tensor(data_list, batch_size):
    """将大型列表分批次转换为Tensor"""
    for i in tqdm(range(0, len(data_list), batch_size), desc="Converting the list to tensor"):
        yield tf.convert_to_tensor(data_list[i:i + batch_size], dtype=tf.float32)

# 定义一个生成器函数来逐个加载.npy文件
def load_batches(directory, file_pattern, num_batches):
    for i in range(num_batches):
        file_path = os.path.join(directory, file_pattern+f'{i}')
        batch_data = np.load(file_path)
        yield batch_data


#%%
# 对所有数据进行拆分以获取训练集和测试集
(user_train_nnum, user_test_nnum,
 user_train_num, user_test_num,
 car_train_nnum, car_test_nnum,
 car_train_num, car_test_num,
 ys_train, ys_test) = train_test_split(user_nnum, user_vecs_num, car_nnum, car_vecs_num, ys, train_size=0.8, shuffle=True, stratify=ys, random_state=2023)

# 对测试集进行进一步拆分以获得验证集和最终的测试集
(user_val_nnum, user_test_nnum,
 user_val_num, user_test_num,
 car_val_nnum, car_test_nnum,
 car_val_num, car_test_num,
 ys_val, ys_test) = train_test_split(user_test_nnum, user_test_num, car_test_nnum, car_test_num, ys_test, train_size=0.5, shuffle=True, stratify=ys_test, random_state=2023)

# 分割文本为适应模型最大长度的多个部分
max_length_comments = 2048  # 模型的最大长度限制
max_length = 2700
# stride = 512  # 可以重叠的token数量
batch_size_user = 4
batch_size = 4

#%%
user_train_emb_list = embeddings_for_features(user_train_nnum, max_length_comments=2048, batch_size_user=4)

#%%
directory = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/'
save_batch_tensor_numpy(user_train_emb_list, tensor_batch_size=1000, directory, file_pattern= 'user_train_emb_tensor_')
# # user_train_emb_list 是一个很大的列表
# tensor_batch_size = 1000  # 选择一个适合系统的批次大小
# tensor_generator = batch_convert_to_tensor(user_train_emb_list, tensor_batch_size)
#
# # 在需要时迭代这个生成器来使用批次的tensor
# for i, tensor in enumerate(tensor_generator):
#     # 处理每个批次的tensor
#     np.save(
#         f'/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/user_train_emb_tensor_{i}.npy',
#         tensor.numpy())

#%%
def save_batch_tensor_numpy(emb_list, tensor_batch_size, directory, file_pattern):

    tensor_generator = batch_convert_to_tensor(emb_list, tensor_batch_size)
    for i, tensor in tqdm(enumerate(tensor_generator), desc="Saving the batch embedding array list"):
        # 处理每个批次的tensor
        np.save(os.path.join(directory, file_pattern+f'{i}'), tensor.numpy())

#%%
def load_batch_numpy_tensor(directory, file_pattern):
    files_in_directory = os.listdir(directory)
    # 使用列表推导式和字符串方法来筛选出符合模式的文件
    pattern = re.compile(rf'^{re.escape(file_pattern)}\d+\.npy$')
    matching_files = [f for f in files_in_directory if pattern.match(f)]

    # 计算匹配文件的数量
    num_batches = len(matching_files)
    accumulated_tensor = None
    for i in tqdm(range(num_batches), desc="Processing accumulated_tensor"):
        ten_a = tf.convert_to_tensor(np.load(os.path.join(directory, file_pattern + f'{i}.npy')), dtype=tf.float32)
        if accumulated_tensor is None:
            accumulated_tensor = ten_a
        else:
            with tf.device('/CPU:0'):
                accumulated_tensor = tf.concat([accumulated_tensor, ten_a], axis=0)

    # all_ar_list = [np.load(os.path.join(directory, file_pattern + f'{i}.npy')) for i in range(num_batches)]
    # all_ar = np.concatenate(all_ar_list, axis=0)
    # # all_tensors = [tf.convert_to_tensor(np.load(os.path.join(directory, file_pattern+f'{i}.npy')), dtype=tf.float32)
    # #                for i in range(num_batches)]
    # accumulated_tensor = None
    # # 逐个合并Tensor
    # for tensor in all_tensors:
    #     if accumulated_tensor is None:
    #         accumulated_tensor = tensor
    #     else:
    #         # 在CPU上进行合并以避免GPU内存不足
    #         with tf.device('/CPU:0'):
    #             accumulated_tensor = tf.concat([accumulated_tensor, tensor], axis=0)

    return accumulated_tensor

#%%
directory = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/'
file_pattern = 'user_train_emb_tensor_'
user_train_emb_tensor = load_batch_numpy_tensor(directory, file_pattern)


#%%

# 假设所有.npy文件都在同一个目录下
directory = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/'
file_pattern = 'user_train_emb_tensor_{}.npy'
num_batches = 9  # 假设你有10个批次的文件

# 创建一个生成器对象
batches = load_batches(directory, file_pattern, num_batches)

# 分批处理数据
for i, batch_data in enumerate(batches):
    # 在这里处理每个批次的数据
    # 例如，你可以直接使用TensorFlow的操作来处理NumPy数组
    tensor = tf.convert_to_tensor(batch_data, dtype=tf.float32)
    # 执行TensorFlow操作...

# 如果你需要最后将所有批次合并为一个Tensor，可以这样做：
all_tensors = [tf.convert_to_tensor(np.load(os.path.join(directory, file_pattern.format(i))), dtype=tf.float32)
               for i in range(num_batches)]
combined_tensor = tf.concat(all_tensors, axis=0)
#%%


#%%
user_val_emb_list = embeddings_for_features(user_val_nnum, max_length_comments=2048, batch_size_user=4)
#%%
directory = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/'
save_batch_tensor_numpy(user_val_emb_list, tensor_batch_size=1000,
                        directory='/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/',
                        file_pattern= 'user_val_emb_tensor_')
#%%
user_val_emb_tensor = load_batch_numpy_tensor(directory, file_pattern='user_val_emb_tensor_')
# np.save('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/user_val_emb_tensor.npy', user_val_emb_tensor.numpy())
#%%
user_test_emb_tensor = embeddings_for_features(user_test_nnum, max_length_comments, batch_size_user)
np.save('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/user_test_emb_tensor.npy', user_test_emb_tensor.numpy())

#%%
# car_train_emb_tensor = embeddings_for_oth_fea(car_train_nnum, max_length=32, batch_size=4)
#%% #join the car features for embeddings

car_train_emb_list = embeddings_for_features(car_train_nnum, max_length, batch_size_user=4)
#%%
 # save the car_train_emb
directory = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/'
save_batch_tensor_numpy(car_train_emb_list, tensor_batch_size=300,
                        directory='/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/',
                        file_pattern= 'car_train_emb_tensor_')
#%%
car_train_emb_tensor = load_batch_numpy_tensor(directory, file_pattern='car_train_emb_tensor_')

#%%
car_val_emb_list = embeddings_for_features(car_val_nnum, max_length, batch_size_user=4)

#%%
directory = '/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/'
save_batch_tensor_numpy(car_val_emb_list, tensor_batch_size=200,
                        directory='/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/',
                        file_pattern= 'car_val_emb_tensor_')
#%%
car_val_emb_tensor = load_batch_numpy_tensor(directory, file_pattern='car_val_emb_tensor_')

#%%
# np.save('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/car_train_emb_tensor.npy', car_train_emb_tensor.numpy())
#%%
# car_val_emb_tensor = embeddings_for_oth_fea(car_val_nnum, max_length, batch_size)
# np.save('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/car_val_emb_tensor.npy', car_val_emb_tensor.numpy())
#%%
# car_test_emb_tensor = embeddings_for_oth_fea(car_test_nnum, max_length, batch_size)
# np.save('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data/car_test_emb_tensor.npy', car_test_emb_tensor.numpy())


#%%

print(car_train_emb_tensor.shape)

print(car_train_emb_tensor.shape)

print(user_train_emb_tensor.shape)


#%% # transfer the numpy features to tensors
user_train_num_tensor = tf.convert_to_tensor(user_train_num, dtype=tf.float32)
car_train_num_tensor = tf.convert_to_tensor(car_train_num, dtype=tf.float32)
ys_train_tensor = tf.convert_to_tensor(ys_train, dtype=tf.float32)

user_val_num_tensor = tf.convert_to_tensor(user_val_num, dtype=tf.float32)
car_val_num_tensor = tf.convert_to_tensor(car_val_num, dtype=tf.float32)
ys_val_tensor = tf.convert_to_tensor(ys_val, dtype=tf.float32)

#%%
user_test_num_tensor = tf.convert_to_tensor(user_test_num, dtype=tf.float32)
car_test_num_tensor = tf.convert_to_tensor(car_test_num, dtype=tf.float32)
ys_test_tensor = tf.convert_to_tensor(ys_test, dtype=tf.float32)


#%%
# 加载numpy数组并转换回张量
user_train_emb_tensor = tf.convert_to_tensor(np.load('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data_4000/user_train_emb_tensor.npy'))

user_val_emb_tensor = tf.convert_to_tensor(np.load('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data_4000/user_val_emb_tensor.npy'))

user_test_emb_tensor = tf.convert_to_tensor(np.load('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data_4000/user_test_emb_tensor.npy'))

car_train_emb_tensor = tf.convert_to_tensor(np.load('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data_4000/car_train_emb_tensor.npy'))

car_val_emb_tensor = tf.convert_to_tensor(np.load('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data_4000/car_val_emb_tensor.npy'))

car_test_emb_tensor = tf.convert_to_tensor(np.load('/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data_4000/car_test_emb_tensor.npy'))

"""# building model"""
#%%
import tensorflow.keras.layers, tensorflow.keras.models

from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Flatten
from tensorflow.keras.models import Model


#%%
num_outputs = 64
uc_NN = tf.keras.models.Sequential([
    # tf.keras.layers.Dense(2048, activation='swish'),
    # tf.keras.layers.Dense(1024, activation='swish'),
    tf.keras.layers.Dense(512, activation='swish'),
    tf.keras.layers.Dense(256, activation='swish'),
    tf.keras.layers.Dense(128, activation='swish'),
    # tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.Dense(num_outputs, activation='swish'),
])
# car_NN = tf.keras.models.Sequential([
#     # tf.keras.layers.Dense(2018, activation='swish'),
#     # tf.keras.layers.Dense(1024, activation='swish'),
#     tf.keras.layers.Dense(512, activation='swish'),
#     tf.keras.layers.Dense(256, activation='swish'),
#     tf.keras.layers.Dense(128, activation='swish'),
#     # tf.keras.layers.Dense(64, activation='swish'),
#     tf.keras.layers.Dense(num_outputs, activation='swish'),
# ])
uc_nn_NN = tf.keras.models.Sequential([
    # tf.keras.layers.Dense(2018, activation='swish'),
    tf.keras.layers.Dense(1024, activation='swish'),
    tf.keras.layers.Dense(512, activation='swish'),
    tf.keras.layers.Dense(256, activation='swish'),
    tf.keras.layers.Dense(128, activation='swish'),
    # tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.Dense(num_outputs, activation='swish'),
])


#%%
# 已经定义了以下变量
uc_train_emb_tensor = tf.concat([user_train_emb_tensor, car_train_emb_tensor],axis=1)
uc_val_emb_tensor = tf.concat([user_val_emb_tensor, car_val_emb_tensor],axis=1)
uc_test_emb_tensor = tf.concat([user_test_emb_tensor, car_test_emb_tensor],axis=1)
num_user_features = user_train_emb_tensor.shape[-1]
num_car_features = car_train_emb_tensor.shape[-1]
num_user_num_features = user_train_num_tensor.shape[-1]
num_car_num_features = car_train_num_tensor.shape[-1]
num_uc_num_features = uc_train_emb_tensor.shape[-1]


# 定义Multi-head 注意力机制层
def multi_head_attention_block(query, key, value, num_heads, key_dim):
    # 使用Keras的MultiHeadAttention层
    mh_attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    attention_output = mh_attention_layer(query, key, value)
    return attention_output

def attention_mechanism(user_tensor, car_tensor):
    """
    Implement a simple dot-product attention between user and car tensors.
    """
    # Compute attention scores
    attention_scores = tf.keras.layers.Dot(axes=1)([user_tensor, car_tensor])

    # Compute attention weights for user tensor
    user_attention_weights = tf.keras.layers.Activation("softmax")(attention_scores)
    user_weighted_tensor = tf.keras.layers.Multiply()([user_tensor, user_attention_weights])

    # Compute attention weights for car tensor
    car_attention_weights = tf.keras.layers.Activation("softmax")(attention_scores)
    car_weighted_tensor = tf.keras.layers.Multiply()([car_tensor, car_attention_weights])

    # Combine the original tensor and the weighted tensor
    user_output = tf.keras.layers.Concatenate()([user_tensor, user_weighted_tensor])
    car_output = tf.keras.layers.Concatenate()([car_tensor, car_weighted_tensor])

    return user_output, car_output

# testing
#%%

# 输入层
# input_user_emb = Input(shape=(None, num_user_features))
# input_car_emb = Input(shape=(None, num_car_features))
input_user_num = Input(shape=(num_user_num_features,))
input_car_num = Input(shape=(num_car_num_features,))
input_uc_emb = Input(shape=(None, num_uc_num_features))

# uc_emb_combined = Concatenate()([input_user_emb, input_car_emb])

# 对嵌入向量应用Multi-head 注意力机制
attention_output = multi_head_attention_block(input_uc_emb, input_uc_emb, input_uc_emb, num_heads=10, key_dim=512)
# car_attention_output = multi_head_attention_block(input_car_emb, input_car_emb, input_car_emb, num_heads=2, key_dim=1024)

# 池化层，将注意力机制的输出降维
uc_vector = GlobalAveragePooling1D()(attention_output)
# car_vector = GlobalAveragePooling1D()(car_attention_output)

# 数值特征和经过注意力机制处理的嵌入向量合并
uc_num_combined = Concatenate()([input_user_num, input_car_num])
# car_combined = Concatenate()([uc_vector, input_car_num])

# 定义神经网络层来进一步处理合并后的特征
# user_nn_output = Dense(256, activation='relu')(user_combined)
# car_nn_output = Dense(256, activation='relu')(car_combined)
uc_nn_output = uc_nn_NN(uc_num_combined)
# car_nn_output = car_NN(car_combined)
uc_output = uc_NN(uc_vector)


# 将用户和车辆的特征合并，以用于最终的预测
vu, vc = attention_mechanism(uc_nn_output, uc_output)

vu = tf.linalg.l2_normalize(vu, axis=1)
vc = tf.linalg.l2_normalize(vc, axis=1)

# d_uv = tf.keras.layers.Dot(axes=1)([vu, vc])
# d_out1 = tf.keras.layers.Dot(axes=1)([vu, uc])
# d_out2 = tf.keras.layers.Dot(axes=1)([vc, uc])
d_out = tf.keras.layers.Dot(axes=1)([vu, vc])
output = tf.keras.layers.Dense(1, activation='sigmoid')(d_out)

# # 预测层
# output = Dense(1, activation='sigmoid')(combined_features)

# 构建模型
model = Model(inputs=[input_uc_emb, input_user_num, input_car_num], outputs=output)

# 显示模型摘要
model.summary()


#%% # 输入层
input_user_emb = Input(shape=(None, num_user_features))
input_car_emb = Input(shape=(None, num_car_features))
input_user_num = Input(shape=(num_user_num_features,))
input_car_num = Input(shape=(num_car_num_features,))

# 对嵌入向量应用Multi-head 注意力机制
user_attention_output = multi_head_attention_block(input_user_emb, input_user_emb, input_user_emb, num_heads=2, key_dim=64)
car_attention_output = multi_head_attention_block(input_car_emb, input_car_emb, input_car_emb, num_heads=2, key_dim=64)

# 池化层，将注意力机制的输出降维
user_vector = GlobalAveragePooling1D()(user_attention_output)
car_vector = GlobalAveragePooling1D()(car_attention_output)

# 数值特征和经过注意力机制处理的嵌入向量合并
user_combined = Concatenate()([user_vector, input_user_num])
car_combined = Concatenate()([car_vector, input_car_num])

# 定义神经网络层来进一步处理合并后的特征
# user_nn_output = Dense(256, activation='relu')(user_combined)
# car_nn_output = Dense(256, activation='relu')(car_combined)
user_nn_output = user_NN(user_combined)
car_nn_output = car_NN(car_combined)


# 将用户和车辆的特征合并，以用于最终的预测
vu, vc = attention_mechanism(user_nn_output, car_nn_output)

vu = tf.linalg.l2_normalize(vu, axis=1)
vc = tf.linalg.l2_normalize(vc, axis=1)

d_out = tf.keras.layers.Dot(axes=1)([vu, vc])
output = tf.keras.layers.Dense(1, activation='sigmoid')(d_out)

# # 预测层
# output = Dense(1, activation='sigmoid')(combined_features)

# 构建模型
model = Model(inputs=[input_user_emb, input_car_emb, input_user_num, input_car_num], outputs=output)

# 显示模型摘要
model.summary()



#%%
tf.keras.utils.plot_model(model, "/Users/depblu/Documents/Gitlab/线下验证数据/E8/model/test_model4_embeddingsmodel/embedding_data_4000/my_model.png")
#%%
def contrastive_loss(y_true, y_pred, margin=1):
    """
    Contrastive loss function.
    - y_true: True labels, 1 for positive pairs, 0 for negative pairs.
    - y_pred: Predicted labels.
    - margin: Contrastive margin.
    """
    # 计算正样本的损失（用户购买车型）
    positive_loss = y_true * tf.square(y_pred)

    # 计算负样本的损失（用户未购买车型）
    negative_loss = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))

    # 计算总损失
    loss = tf.reduce_mean(positive_loss + negative_loss)
    return loss

#%%
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=opt, loss=cost_fn)

# contrastive_loss

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=60, # number of epochs to wait
                                                  restore_best_weights=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', # monitor the validation loss
    factor=0.5,         # reduce the learning rate to 10% of its current value
    patience=30,         # reduce the learning rate if the metric does not improve for 5 consecutive epochs
    min_lr=1e-7,        # set the minimum learning rate
    verbose=1           # print messages about learning rate reduction
)

# 创建ModelCheckpoint回调
checkpoint_filepath = '/content/best_model2.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_weights_only=False # 保存整个模型。如果只想保存权重，设置为True。
)

#%%
# model.fit([user_train, car_train], ys_train, epochs=12)
history = model.fit([uc_train_emb_tensor, user_train_num_tensor, car_train_num_tensor], ys_train_tensor,
                    epochs=100,
                    validation_data=([uc_test_emb_tensor, user_test_num_tensor, car_test_num_tensor], ys_test_tensor),
                    callbacks = [model_checkpoint_callback, early_stopping, reduce_lr_callback])
#%%
# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()














