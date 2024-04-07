import pandas as pd
from paddlenlp.transformers import SkepTokenizer, SkepModel, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
# from paddle.fluid.framework import  _non_static_mode
# from paddle.fluid.data_feeder import check_variable_and_dtype
import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
import configparser
config = configparser.ConfigParser()
config.read("config.config", encoding="utf-8")

Model_name=config.get('Settings','Model_name')
if 'uncased' in Model_name or 'bert-base-chinese' in Model_name:
    tokenizer = BertTokenizer.from_pretrained(Model_name)
elif 'skep' in Model_name:
    tokenizer = SkepTokenizer.from_pretrained(Model_name)

def file_read_meituan(path,mode='train'):
    data=pd.read_csv(path,nrows=None)
    # 把3到21列的每行数据合并到一起，作为label
    # 在推理阶段，只需要reviews和key_words
    # 返回格式：reviews: [review1,review2,...],labels:[[label1,label2,...],...],key_words:[[key_word1,key_word2],...]
    if mode == 'train':
        for i in range(len(data)):
            k=[keyword.split('#') for keyword in data.iloc[i,-1].split('@')]
            k=[j for i in k for j in i]
            if len(k)>1:
                reviews=data.iloc[i,1]
                labels=data.iloc[i,3:21].values.tolist()
                # 把label里的-1，0，1转为1
                labels=[j+2 if j in [-1,0,1]  else j for j in labels]
                # 把label里的-2转为0
                labels=[0 if j==-2 else j for j in labels]
                key_words=k
                yield  {'reviews':reviews,'senti_labels':labels,'key_words':key_words}
    else:
        for i in range(len(data)):
            k=[''.join(keyword.split('#')) for keyword in data.iloc[i,2].split('@')]
            if len(k)>1:
                reviews=data.iloc[i,1]
                key_words=k
                yield  {'reviews':reviews,'key_words':key_words}

def file_read_semeval(path,clean=True,root=config.get('Settings','root'),sep=','):
    print('读取文件路径：',path)
    data=pd.read_csv(path,nrows=None,sep=sep)
    label2id={val:i for i, val in enumerate(open(root+'label.txt', 'r', encoding='utf-8').read().splitlines())}
    dictt={}
    for i in range(len(data)):
        if clean ==True and data['key_words'][i] == 'NEG':
            continue
        try:
            data['review'][i]=data['review'][i].lower()
        except:
            pass

        if 'SEMEVAL' in root or 'MAMS' in root:                                              # 英文数据集
            if data['review'][i] not in dictt:                                               # 如果该评论不在字典中
                listt= [0] * len(label2id)                                                   # 初始化一个全0的列表，作为标签
                listt[label2id[data['labels'][i]]]=int(data['key_words'][i].split(' ')[0])   # key_words的第一个词是情感类别，第二个词是情感强度
                dictt.update({data['review'][i]:[                                            # 更新字典，key是评论
                                                 [data['key_words'][i].split(' ')[1:][0].lower()],        # 关键词是字典第一个元素
                                                 listt                                       # 标签是字典第二个元素
                                                 ]})
            else:                                                                            # 如果该评论在字典中
                dictt[data['review'][i]][0].append(data['key_words'][i].split(' ')[1:][0].lower())   # 更新字典，key是评论，字典第一个元素，关键词列表，添加一个关键词
                dictt[data['review'][i]][1][label2id[data['labels'][i]]]=int(data['key_words'][i].split(' ')[0])  # 更新字典，key是评论，字典第二个元素，标签列表，更新标签
                dictt[data['review'][i]][0]=list(set(dictt[data['review'][i]][0]))           # 关键词要去重

        else:                                                                                # 中文数据集
            if data['review'][i] not in dictt:                                               # 如果该评论不在字典中
                listt= [0] * len(label2id)
                if int(data['labels'][i]) <=11:
                    listt[int(data['labels'][i])]=1
                elif int(data['labels'][i]) <=23:
                    listt[int(data['labels'][i])-12]=2
                else:
                    listt[int(data['labels'][i])-24]=3                                       # 更新标签的情感强度
                dictt.update({data['review'][i]:[                                            # 更新字典，key是评论
                                                 [data['key_words'][i]],                     # 关键词是字典第一个元素
                                                 listt                                       # 标签是字典第二个元素
                                                 ]})
            else:                                                                            # 如果该评论在字典中
                dictt[data['review'][i]][0].append(data['key_words'][i])                     # 更新字典，key是评论，字典第一个元素，关键词列表，添加一个关键词
                if int(data['labels'][i]) <=11:
                    listt[int(data['labels'][i])]=1
                elif int(data['labels'][i]) <=23:
                    listt[int(data['labels'][i])-12]=2
                else:
                    listt[int(data['labels'][i])-24]=3                                       # 更新字典，key是评论，字典第二个元素，标签列表，更新标签
                dictt[data['review'][i]][0]=list(set(dictt[data['review'][i]][0]))           # 关键词要去重
        reviews=[each for each in dictt.keys()]

    for i in range(len(dictt)):
        try:
            yield  {'reviews':reviews[i],
                    'key_words':list(set(dictt[reviews[i]][0])),
                    'senti_labels':dictt[reviews[i]][1]}
        except:
            continue
# {'reviews': "i've also been amazed at all the new additions in the past few years: a new jazz bar, the most fantastic dining garden, the best thin crust pizzas, and now a lasagna menu which is to die for (these are not your average lasagnas)!",
# 'key_words': ['Jazz', 'Dining', 'Garden', 'Lasagna', 'Thin'],
# 'senti_labels': [0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3]}

def convert_example_to_feature_normal(example,
                                      tokenizer=tokenizer,
                                      max_seq_len=config.getint('Settings','Max_seq_len'),
                                      q_length=config.getint('Settings','Q_length')):

    if 'SEMEVAL' in config.get('Settings','Root'):
        key_words = ' '.join(example["key_words"])
    else:
        key_words = example["key_words"][0]
    encoded_inputs_data = tokenizer(example["reviews"],max_length=max_seq_len,truncation=True,padding='max_length')
    encoded_inputs_term = tokenizer(key_words,         max_length=q_length,   truncation=True,padding='max_length')
    encoded_inputs_data['attention_mask']=[1 if i!=tokenizer.pad_token_id else 0 for i in encoded_inputs_data['input_ids']]
    encoded_inputs_term['attention_mask']=[1 if i!=tokenizer.pad_token_id else 0 for i in encoded_inputs_term['input_ids']]

    return encoded_inputs_data['input_ids'],\
        encoded_inputs_term['input_ids'], \
        encoded_inputs_data['attention_mask'], \
        encoded_inputs_term['attention_mask'], \
        example['senti_labels']

def convert_example_to_feature_special(example,
                                       tokenizer=BertTokenizer.from_pretrained(config.get('Settings','Model_name')),
                                       max_seq_len=config.getint('Settings','Max_seq_len'),
                                       q_length=config.getint('Settings','Q_length')):
    if 'SEMEVAL' in config.get('Settings','Root'):
        key_words = ' '.join(example["key_words"])
    else:
        key_words = example["key_words"][0]
    encoded_inputs_data = tokenizer(example["reviews"],max_length=max_seq_len,truncation=True,padding='max_length')
    encoded_inputs_term = tokenizer(key_words,         max_length=q_length,   truncation=True,padding='max_length')
    encoded_inputs_data['attention_mask']=[1 if i!=tokenizer.pad_token_id else 0 for i in encoded_inputs_data['input_ids']]
    encoded_inputs_term['attention_mask']=[1 if i!=tokenizer.pad_token_id else 0 for i in encoded_inputs_term['input_ids']]

    return encoded_inputs_data['input_ids'],\
        encoded_inputs_term['input_ids'], \
        encoded_inputs_data['attention_mask'], \
        encoded_inputs_term['attention_mask'], \
        example['senti_labels']
# input_ids_data:tensor([  101,  1045,  1005,  2310,  2036,  2042,  3815,  2012,  2035,  1996])  data
# input_ids_term:tensor([  101,  1045,  1005,  2310,  2036,  2042,  3815,  2012,  2035,  1996])  term
# attention_mask_data:tensor([  1, 1,  1,  1,  1,  1,  1,  1, 1,  1])  data
# attention_mask_term:tensor([  1, 1,  1,  1,  1,  1,  1,  1, 1,  1])  term
# senti_labels:tensor([0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3])

def create_dataloader(tokenizer,read_func,convert_func,root,batch_size,name_list,pinpai=None,pred_only=False):

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Stack()
    ): fn(samples)

    if pred_only:
        data = pd.read_csv(root + '关键词抽取结果/关键词抽取结果_' + pinpai + '.tsv', encoding='utf-8', header=None, sep = '\t')
        data['labels'] = [1] * len(data)
        data.columns = ['key_words','review','labels']
        data.to_csv(root + '关键词抽取结果/关键词抽取结果_' + pinpai + '_temp.tsv', encoding='utf-8', index=False, sep = '\t')
        pred_ds = load_dataset(read_func, path=root + '关键词抽取结果/关键词抽取结果_' + pinpai + '_temp.tsv', lazy=False, sep ='\t')
        pred_ds = pred_ds.map(convert_func)
        pred_batch_sampler = paddle.io.DistributedBatchSampler(pred_ds, batch_size=batch_size, shuffle=False)
        pred_loader = paddle.io.DataLoader(pred_ds, batch_sampler=pred_batch_sampler, collate_fn=batchify_fn)
        return pred_loader

    else:
        train_ds = load_dataset(read_func, path=root + '关键词抽取结果_'+name_list[0]+'.csv', lazy=False)
        train_ds = train_ds.map(convert_func)
        train_batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=batch_size, shuffle=True)
        train_loader = paddle.io.DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn)

        dev_ds = load_dataset(read_func, path=root + '关键词抽取结果_'+name_list[1]+'.csv', lazy=False)
        dev_ds = dev_ds.map(convert_func)
        dev_batch_sampler = paddle.io.DistributedBatchSampler(dev_ds, batch_size=batch_size, shuffle=True)
        dev_loader = paddle.io.DataLoader(dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn)

        test_ds = load_dataset(read_func, path=root + '关键词抽取结果_'+name_list[2]+'.csv', lazy=False)
        test_ds = test_ds.map(convert_func)
        test_batch_sampler = paddle.io.DistributedBatchSampler(test_ds, batch_size=batch_size, shuffle=False)
        test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn)

        pred_ds = load_dataset(read_func, path=root + '关键词抽取结果_'+name_list[3]+'.csv', lazy=False)
        pred_ds = pred_ds.map(convert_func)
        pred_batch_sampler = paddle.io.DistributedBatchSampler(pred_ds, batch_size=batch_size, shuffle=False)
        pred_loader = paddle.io.DataLoader(pred_ds, batch_sampler=pred_batch_sampler, collate_fn=batchify_fn)

        return train_loader,dev_loader,test_loader,pred_loader
# train_loader,dev_loader,test_loader,pred_loader


def multi_margin_loss(inputs,
                      labels,
                      p: int = 1,
                      margin: float = 1.0,
                      weight=None,
                      reduction='mean',
                      name=None):

    input = paddle.reshape(inputs, shape=(-1, inputs.shape[-1]))
    label = paddle.reshape(labels, shape=(-1, labels.shape[-1]))
    label = paddle.argmax(label, axis=1)
    label = label.reshape((-1, 1))
    index_sample = paddle.index_sample(input, label)

    loss = paddle.mean(paddle.pow(
            paddle.clip(margin - index_sample + input, min=0.0), p),
                           axis=1) - margin**p / paddle.shape(input)[1]
    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss
