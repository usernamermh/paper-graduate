import paddle
from paddlenlp.transformers import WordEmbedding
from paddle.nn.layer.transformer import MultiHeadAttention
import paddle.nn as nn
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report
from functions import multi_margin_loss
from paddle.nn import LSTM
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

class Bert_LSTM_Attention(nn.Layer):
    def __init__(self, model,hidden_size,num_classes,num_attention_heads,num_sentis,batch_func,lstm,res):
        super(Bert_LSTM_Attention, self).__init__()
        self.model = model
        self.attention = MultiHeadAttention(hidden_size,num_attention_heads,dropout=0.1,need_weights=True)
        self.l = nn.Linear(hidden_size, num_classes*(num_sentis+1))
        self.batch_func = batch_func
        self.res = res
        self.layer_norm = nn.LayerNorm(hidden_size)
        if lstm == 'Lstm':
            self.lstm_flag = True
            self.lstm = LSTM(input_size=hidden_size,hidden_size=int(hidden_size/2),num_layers=2, direction='bidirectional')
        else:
            self.lstm_flag = False
        
    def forward(self, input_ids_data, input_ids_term, attention_mask_input_data, attention_mask_input_term, Max_seq_len, is_training=True):
        outputs_bert_data = self.model(input_ids_data)[0]
        outputs_bert_term = self.model(input_ids_term)[0]
        if self.lstm_flag:
            outputs_bert_data, (last_hidden, last_cell) = self.lstm(outputs_bert_data)
            outputs_bert_term, (last_hidden, last_cell) = self.lstm(outputs_bert_term)
        
        outputs_attention,cross_attention = self.attention(outputs_bert_term,outputs_bert_data,outputs_bert_data,None,None) # outputs_attention [16,20,1024]
        
        if self.res == 'Res':
            outputs_attention = outputs_attention + outputs_bert_term
            outputs_attention = self.layer_norm(outputs_attention)
        outputs_attention = paddle.nn.functional.dropout(outputs_attention,p=0.1)

        if self.batch_func == 'max':
            outputs_attention_pool = paddle.nn.functional.max_pool1d(outputs_attention.transpose([0, 2, 1]), kernel_size=Max_seq_len, stride=Max_seq_len, padding=0)
        elif self.batch_func == 'mean':
            outputs_attention_pool = paddle.nn.functional.avg_pool1d(outputs_attention.transpose([0, 2, 1]), kernel_size=Max_seq_len, stride=Max_seq_len, padding=0)
        elif self.batch_func == 'cls':
            outputs_attention_pool = paddle.unsqueeze(outputs_attention[:,0,:], 2)
        
        # outputs_attention_pool [Batchsize,Max_seq_len,1]
        state = self.l(paddle.squeeze(outputs_attention_pool,-1))
        return state,outputs_bert_data,cross_attention

def train_bert_attention(Model, train_loader, num_epoch, writer, lr_scheduler, optimizer, checkpoint, log_step, max_seqlength, num_sentis, loss_func):
    Model.train()
    global_step = 0
    print(checkpoint)
    for epoch in range(1, num_epoch + 1):
        for batch_data in train_loader():
            input_ids_data, input_ids_term, attention_mask_input_data, attention_mask_input_term, senti_labels = batch_data

            # ----------------------------------------注释掉-----------------------------------------
            # labels = []
            # if num_sentis == 3:
            #     for i in range(0, len(senti_labels)):
            #         l = []
            #         for each_label in senti_labels[i]:
            #             if each_label == paddle.to_tensor(0, dtype='int32'):
            #                 l.append([1, 0, 0, 0])
            #             elif each_label == paddle.to_tensor(1, dtype='int32'):
            #                 l.append([0, 1, 0, 0])
            #             elif each_label == paddle.to_tensor(2, dtype='int32'):
            #                 l.append([0, 0, 1, 0])
            #             elif each_label == paddle.to_tensor(3, dtype='int32'):
            #                 l.append([0, 0, 0, 1])
            #         labels.append(l)
            # elif num_sentis == 4:
            #     for i in range(0, len(senti_labels)):
            #         l = []
            #         for each_label in senti_labels[i]:
            #             if each_label == paddle.to_tensor(0, dtype='int32'):
            #                 l.append([1, 0, 0, 0, 0])
            #             elif each_label == paddle.to_tensor(1, dtype='int32'):
            #                 l.append([0, 1, 0, 0, 0])
            #             elif each_label == paddle.to_tensor(2, dtype='int32'):
            #                 l.append([0, 0, 1, 0, 0])
            #             elif each_label == paddle.to_tensor(3, dtype='int32'):
            #                 l.append([0, 0, 0, 1, 0])
            #             elif each_label == paddle.to_tensor(4, dtype='int32'):
            #                 l.append([0, 0, 0, 0, 1])
            #         labels.append(l)
            # labels = paddle.to_tensor(labels, dtype='int32').cuda()
            # ----------------------------------------注释掉-----------------------------------------


            state, outputs_bert, cross_attention = Model(input_ids_data,
                                                          input_ids_term,
                                                          attention_mask_input_data,
                                                          attention_mask_input_term,
                                                          max_seqlength)

            logits = paddle.reshape(state, [len(state), -1, num_sentis + 1])
            logits = paddle.nn.functional.softmax(logits, axis=-1)

            loss = paddle.nn.functional.cross_entropy(logits, senti_labels, soft_label=False, use_softmax=False, reduction='mean')

            loss.backward()
            loss = float(loss)
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)

            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{len(train_loader) * num_epoch} - loss:{loss:.6f}")
            global_step += 1

    paddle.save(Model.state_dict(), f"{checkpoint}/best_cls_bert_attention.pdparams")

def test_bert_attention(Model, test_loader,  max_seqlength, checkpoint, num_sentis, root):
    Model.set_state_dict(paddle.load(f"{checkpoint}/best_cls_bert_attention.pdparams"))
    Model.eval()
    id2label = {i:val for i, val in enumerate(open(root+'label.txt', 'r', encoding='utf-8').read().splitlines())}
    with paddle.no_grad():
        Labels_3=[];Preds_3=[]  # 用于情感分类
        Labels_4=[];Preds_4=[]  # 用于维度分类
        Labels_5=[];Preds_5=[]  # 用于维度-情感分类
        S_G=0;len_S=0;len_G=0   # 用于维度-情感分类
        S_G2=0;len_S2=0;len_G2=0   # 用于维度-情感分类

        for batch_data in tqdm(test_loader()):
            input_ids_data, input_ids_term, attention_mask_input_data, attention_mask_input_term, senti_labels = batch_data
            state, outputs_bert, cross_attention = Model(input_ids_data,
                                                                  input_ids_term,
                                                                  attention_mask_input_data,
                                                                  attention_mask_input_term,
                                                                  max_seqlength)
            # 把模型输出转为独热的形式
            state2_ = paddle.reshape(state, [len(state), -1, num_sentis + 1])
            state2_ = paddle.nn.functional.softmax(state2_, axis=-1)
            # 对于每一个样本
            for each_sample in range(0,len(state2_)):

                state2_pred_3 = [];labels_3=[]  # 用于情感分类
                state2_pred_4 = [];labels_4=[]  # 用于维度分类
                state2_pred_5 = [];labels_5=[]  # 用于维度-情感分类

                # 对于每一个标签
                for i in range(0,len(state2_[each_sample])):
                    # 处理维度-情感分类，和维度分类
                    state = state2_[each_sample][i]
                        
                    # 对于预测值的每一个维度，如果不是0，就把维度和情感值拼接起来
                    if int(paddle.argmax(state)) > 0:
                        state2_pred_5.append(str(int(paddle.argmax(state)))+'_'+id2label[i])
                        state2_pred_4.append(id2label[i])
                        len_S+=1;len_S2+=1

                    # 对于标签的每一个维度，如果不是0，就把维度和情感值拼接起来
                    if senti_labels[each_sample][i] > 0:
                        labels_5.append(str(int(senti_labels[each_sample][i]))+'_'+id2label[i])
                        labels_4.append(id2label[i])
                        len_G+=1;len_G2+=1

                    # 处理情感分类，如果预测和标签的维度都不是0，就把情感加入到预测值列表中
                    state2_pred_3.append(paddle.argmax(state)) if paddle.argmax(state) != 0 and senti_labels[each_sample][i] != 0 else None
                    labels_3.append(senti_labels[each_sample][i]) if paddle.argmax(state) != 0 and senti_labels[each_sample][i] != 0 else None

                Preds_3.extend(state2_pred_3);Labels_3.extend(labels_3)
                Preds_4.append(state2_pred_4);Labels_4.append(labels_4)
                Preds_5.append(state2_pred_5);Labels_5.append(labels_5)

        for each_sample in range(0,len(Preds_5)):
            interlist = list(set(Preds_5[each_sample]).intersection(set(Labels_5[each_sample])))
            for i in range(0,len(interlist)):
                S_G+=1

        for each_sample in range(0,len(Preds_4)):
            interlist = list(set(Preds_4[each_sample]).intersection(set(Labels_4[each_sample])))
            for i in range(0,len(interlist)):
                S_G2+=1

        P=S_G/len_S
        R=S_G/len_G
        F1=2*P*R/(P+R) if P+R != 0 else 0

        P2=S_G2/len_S2
        R2=S_G2/len_G2
        F12=2*P2*R2/(P2+R2) if P2+R2 != 0 else 0
        print(checkpoint)
        print('维度-情感分类效果\n','P:',str(P)[0:5],'R:',str(R)[0:5],'F1:',str(F1)[0:5],'S_G:',str(S_G)[0:5],'len_S:',len_S,'len_G:',len_G)
        print('维度-分类效果\n','P:',str(P2)[0:5],'R:',str(R2)[0:5],'F1:',str(F12)[0:5],'S_G:',str(S_G2)[0:5],'len_S:',len_S2,'len_G:',len_G2)
        print('情感-分类效果',classification_report(Labels_3, Preds_3, digits=4),sep='\n')

        data = np.zeros([3,3])
        for i in range(0,len(Labels_3)):
            data[Labels_3[i]-1,Preds_3[i]-1] += 1


def pred_bert_attention_for_one_case(Model, tokenizer, sentence, max_seq_len, checkpoint, num_sentis, root):
    Model.set_state_dict(paddle.load(f"{checkpoint}/best_cls_bert_attention.pdparams"))
    Model.eval()

    id2label = {i: val for i, val in enumerate(open(root + 'label.txt', 'r', encoding='utf-8').read().splitlines())}

    encoded_inputs_data = tokenizer(sentence.split('\t')[1],max_length=max_seq_len,truncation=True,padding='max_length')
    encoded_inputs_term = tokenizer(sentence.split('\t')[0],max_length=max_seq_len,truncation=True,padding='max_length')
    encoded_inputs_data['attention_mask']=[1 if i!=tokenizer.pad_token_id else 0 for i in encoded_inputs_data['input_ids']]
    encoded_inputs_term['attention_mask']=[1 if i!=tokenizer.pad_token_id else 0 for i in encoded_inputs_term['input_ids']]

    with paddle.no_grad():
        input_ids_data = paddle.to_tensor([encoded_inputs_data['input_ids']])
        input_ids_term = paddle.to_tensor([encoded_inputs_term['input_ids']])
        attention_mask_input_data = paddle.to_tensor([encoded_inputs_data['attention_mask']])
        attention_mask_input_term = paddle.to_tensor([encoded_inputs_term['attention_mask']])

        state, outputs_bert, cross_attention = Model(input_ids_data,
                                                     input_ids_term,
                                                     attention_mask_input_data,
                                                     attention_mask_input_term,
                                                     max_seq_len)

        state2_ = paddle.reshape(state, [len(state), -1, num_sentis + 1])
        state2_ = paddle.nn.functional.softmax(state2_, axis=-1)
        state2_ = state2_[0]

        for i in range(0, len(state2_)):
            state = state2_[i]
            if int(paddle.argmax(state)) > 0:
                print(id2label[i]+str(int(paddle.argmax(state))))

    # pred_bert_attention_for_one_case(Model, Tokenizer, '收到速度快\t11月19日下单，今天21日下午收到衣服，京东速度很快', Max_seq_len, Checkpoint, Num_sentis, Root)

def pred_bert_attention(Model, pred_loader, max_seqlength, checkpoint, num_sentis, root, pinpai):
    data_path_read = root + '关键词抽取结果/关键词抽取结果_' + pinpai + '.tsv'
    data_path_save = root + '维度-情感分类结果/维度-情感分类结果_' + pinpai + '.tsv'

    print('推理数据读取路径:', data_path_read)
    print('推理数据写入路径:', data_path_save)

    Model.set_state_dict(paddle.load(f"{checkpoint}/best_cls_bert_attention_69.pdparams"))
    Model.eval()
    id2label_cls = {i: val for i, val in enumerate(open(root + 'label.txt', 'r', encoding='utf-8').read().splitlines())}

    Data = pd.read_csv(data_path_read, encoding='utf-8', sep='\t')
    Data.columns = ['key_words', 'review']
    sentence_list = list(set(Data['review'].tolist()))

    with open(data_path_save, 'w', encoding='utf-8') as f:
        with paddle.no_grad():
            index = 0
            for batch_data in tqdm(pred_loader()):
                input_ids_data, input_ids_term, attention_mask_input_data, attention_mask_input_term, senti_labels = batch_data
                states, outputs_bert, cross_attention = Model(input_ids_data,
                                                              input_ids_term,
                                                              attention_mask_input_data,
                                                              attention_mask_input_term,
                                                              max_seqlength)
                states = paddle.reshape(states, [len(states), -1, num_sentis + 1])
                states = paddle.nn.functional.softmax(states, axis=-1)
                # 对于每一个样本
                for each_sample in range(0, len(states)):

                    cats = []  # 维度-情感类别的id
                    probs = []  # 某一维度各情感的概率值

                    for i in range(0, len(states[each_sample])):
                        # 对于每一个维度
                        state = states[each_sample][i]

                        # 如果是中性情感,存储各情感概率
                        if paddle.argmax(state) == 2:
                            l = paddle.flatten(paddle.to_tensor([each for each in state[1:]]))

                        # 如果是极性情感
                        if paddle.argmax(state) == 1 or paddle.argmax(state) == 3:
                            l = paddle.to_tensor([0] * 3, dtype='float32')
                            l[paddle.argmax(state[1:])] = float(1)

                        # 如果不涉及该情感，则放弃
                        if paddle.argmax(state) == 0:
                            l = None

                        # 如果存在情感概率
                        if l is not None:
                            cats.append(int(paddle.argmax(l)) * len(id2label_cls) + i)
                            probs.append([round(each, 2) for each in l.tolist()])

                    if not cats:
                        # 选择第0个索引值最小的维度
                        id_ = paddle.argmin(paddle.argmin(states[each_sample], axis=1))

                        # 对后三个情感值做softmax
                        state = states[each_sample][id_][1:].tolist()
                        state = paddle.nn.functional.softmax(paddle.to_tensor([each / sum(state) for each in state]))

                        # 如果是中性情感,l存储各情感概率
                        if paddle.argmax(state) == 1:
                            l = paddle.to_tensor([each for each in state])

                        # 如果是极性情感
                        if paddle.argmax(state) == 0 or paddle.argmax(state) == 2:
                            l = paddle.to_tensor([0] * 3, dtype='float32')
                            l[paddle.argmax(state)] = float(1)

                        cats.append(int(paddle.argmax(l)) * len(id2label_cls) + i)
                        l = paddle.flatten(l).tolist()
                        l = [round(each, 2) for each in l]
                        probs.append(l)

                    l = ','.join([str(int(each)) for each in cats])
                    p = [str(each).replace('[', '').replace(']', '') for each in probs]
                    p = '|'.join(p)
                    f.write(l + '\t' + p + '\t' + sentence_list[index] + '\n')
                    index += 1

    # for pinpai in ['ALBD','AND','AT','BNL','GRN','LN','QPL','TB']:
    #     Pred_loader = create_dataloader(Tokenizer,read_func,Convert_func,Root,Batch_size,None,pinpai,pred_only=True)
    #     pred_bert_attention(Model, Pred_loader, Max_seq_len, Checkpoint, Num_sentis, Root, pinpai, mode='traditional')
