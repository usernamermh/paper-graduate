import paddle
import paddle.nn as nn
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report
from functions import multi_margin_loss
from paddle.nn import LSTM


class Bert_LSTM(nn.Layer):
    def __init__(self,model,hidden_size,num_sentis,num_classes):
        super(Bert_LSTM,self).__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.lstm = LSTM(input_size=hidden_size, hidden_size=int(hidden_size / 2), num_layers=1,direction='bidirectional',time_major=False)
        self.l = nn.Linear(hidden_size, num_classes*(num_sentis+1))

    def forward(self, input_ids1, input_ids2):
        embedding1 = self.model(input_ids1)[0]
        embedding2 = self.model(input_ids2)[0]
        rnn_out, (last_hidden1, last_cell)  = self.lstm(embedding1)
        rnn_out, (last_hidden2, last_cell)  = self.lstm(embedding2)
        last_hidden1 = paddle.nn.functional.dropout(last_hidden1)
        last_hidden2 = paddle.nn.functional.dropout(last_hidden2)
        last_hidden1 = paddle.transpose(last_hidden1,perm=[1,0,2])
        last_hidden2 = paddle.transpose(last_hidden2,perm=[1,0,2])
        last_hidden1 = paddle.reshape(last_hidden1,[-1,self.hidden_size])
        last_hidden2 = paddle.reshape(last_hidden2,[-1,self.hidden_size])
        last_hidden = paddle.mean(paddle.stack([last_hidden1,last_hidden2]),axis=0)
        last_hidden = self.l(last_hidden)
        return last_hidden

def train_bert_lstm(Model, train_loader, num_epoch, writer, lr_scheduler, optimizer, checkpoint, log_step, num_sentis, loss_func):
    Model.train()
    global_step = 0
    print(checkpoint)
    for epoch in range(1, num_epoch + 1):
        for batch_data in train_loader():
            input_ids_data, input_ids_term, attention_mask_input_data, attention_mask_input_term, senti_labels = batch_data
            labels = []
            if num_sentis == 3:
                for i in range(0, len(senti_labels)):
                    l = []
                    for each_label in senti_labels[i]:
                        if each_label == paddle.to_tensor(0, dtype='int32'):
                            l.append([1, 0, 0, 0])
                        elif each_label == paddle.to_tensor(1, dtype='int32'):
                            l.append([0, 1, 0, 0])
                        elif each_label == paddle.to_tensor(2, dtype='int32'):
                            l.append([0, 0, 1, 0])
                        elif each_label == paddle.to_tensor(3, dtype='int32'):
                            l.append([0, 0, 0, 1])
                    labels.append(l)
            elif num_sentis == 4:
                for i in range(0, len(senti_labels)):
                    l = []
                    for each_label in senti_labels[i]:
                        if each_label == paddle.to_tensor(0, dtype='int32'):
                            l.append([1, 0, 0, 0, 0])
                        elif each_label == paddle.to_tensor(1, dtype='int32'):
                            l.append([0, 1, 0, 0, 0])
                        elif each_label == paddle.to_tensor(2, dtype='int32'):
                            l.append([0, 0, 1, 0, 0])
                        elif each_label == paddle.to_tensor(3, dtype='int32'):
                            l.append([0, 0, 0, 1, 0])
                        elif each_label == paddle.to_tensor(4, dtype='int32'):
                            l.append([0, 0, 0, 0, 1])
                    labels.append(l)
            labels = paddle.to_tensor(labels, dtype='int32').cuda()

            state = Model(input_ids_data, input_ids_term)

            logits = paddle.reshape(state, [len(state), -1, num_sentis + 1])
            logits = paddle.nn.functional.softmax(logits, axis=-1)
            labels = paddle.to_tensor(labels, dtype='float32')

            if loss_func == 'cross_entropy':
                loss = paddle.nn.functional.cross_entropy(logits, labels, soft_label=True, use_softmax=False,
                                                          reduction='mean')
            elif loss_func == 'margin':
                loss = multi_margin_loss(logits, labels, reduction='mean')
            elif loss_func == 'focal':
                loss = paddle.nn.functional.sigmoid_focal_loss(logits, labels, reduction='mean')

            loss.backward()
            loss = float(loss)
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)

            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{len(train_loader) * num_epoch} - loss:{loss:.6f}")
                paddle.save(Model.state_dict(), f"{checkpoint}/best_cls_bert_lstm.pdparams")
            global_step += 1

    paddle.save(Model.state_dict(), f"{checkpoint}/best_cls_bert_lstm.pdparams")

def test_bert_lstm(Model, test_loader,  max_seqlength, checkpoint, num_sentis, root):
    Model.eval()
    Model.set_state_dict(paddle.load(f"{checkpoint}/best_cls_bert_lstm.pdparams"))
    id2label = {i:val for i, val in enumerate(open(root+'label.txt', 'r', encoding='utf-8').read().splitlines())}

    with paddle.no_grad():
        Labels_3=[];Preds_3=[]  # 用于情感分类
        Labels_4=[];Preds_4=[]  # 用于维度分类
        Labels_5=[];Preds_5=[]  # 用于维度-情感分类
        S_G=0;len_S=0;len_G=0   # 用于维度-情感分类
        S_G2=0;len_S2=0;len_G2=0   # 用于维度-情感分类
        for batch_data in test_loader():
            input_ids_data, input_ids_term, attention_mask_input_data, attention_mask_input_term, senti_labels = batch_data
            state, outputs_bert, cross_attention = Model(input_ids_data,
                                                                  input_ids_term,
                                                                  attention_mask_input_data,
                                                                  attention_mask_input_term,
                                                                  max_seqlength)
            # 把模型输出转为独热的形式
            state2_ = paddle.reshape(state, [len(state), -1, num_sentis + 1])
            # 对于每一个样本
            for each_sample in range(0,len(state2_)):

                state2_pred_3 = [];labels_3=[]  # 用于情感分类
                state2_pred_4 = [];labels_4=[]  # 用于维度分类
                state2_pred_5 = [];labels_5=[]  # 用于维度-情感分类

                # 对于每一个标签
                for i in range(0,len(state2_[each_sample])):
                    # softmax一下
                    state = paddle.nn.functional.softmax(state2_[each_sample][i],axis=-1)
                    # 处理维度-情感分类,和维度分类
                    # 对于预测值的每一个维度，如果不是0，就把维度和情感值拼接起来
                    if int(paddle.argmax(state)) != 0:
                        state2_pred_5.append(str(int(paddle.argmax(state)))+'_'+id2label[i])
                        state2_pred_4.append(id2label[i])
                        len_S+=1;len_S2+=1

                    # 对于标签的每一个维度，如果不是0，就把维度和情感值拼接起来
                    if senti_labels[each_sample][i] > 0:
                        labels_5.append(str(int(senti_labels[each_sample][i]))+'_'+id2label[i])
                        labels_4.append(id2label[i])
                        len_G+=1;len_G2+=1

                    # 处理情感分类
                    # 如果预测和标签的维度都不是0，就把情感加入到预测值列表中
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