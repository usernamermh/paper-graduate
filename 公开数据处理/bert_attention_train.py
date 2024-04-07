import warnings
from functions import *
from bert_attention import *
from bert_bare import *
from bert_lstm import *
import configparser
import os
from visualdl import LogWriter
from paddlenlp.transformers import SkepTokenizer, SkepModel, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel

try:
    import seaborn as sns
except:
    pass

warnings.filterwarnings("ignore")
paddle.set_device("gpu:0")

config = configparser.ConfigParser()
config.read("config.config", encoding="utf-8")

Root = config.get('Settings', 'Root')
Num_attention_heads = int(config.get('Settings', 'Num_attention_heads'))
Num_epoch = int(config.get('Settings', 'Num_epoch'))
Batch_size = int(config.get('Settings', 'Batch_size'))
if '2014' in Root:
    Num_sentis = 4
else:
    Num_sentis = 3
Convert = config.get('Settings', 'Convert')
Max_seq_len = int(config.get('Settings', 'Max_seq_len'))
Q_length = int(config.get('Settings', 'Q_length'))
Learning_rate = float(config.get('Settings', 'Learning_rate'))
Model_name = config.get('Settings', 'Model_name')
Lstm = config.get('Settings', 'Lstm')
Res = config.get('Settings', 'Res')
Batch_func = config.get('Settings', 'Batch_func')
Loss_func = config.get('Settings', 'Loss_func')

label2id = {val: i for i, val in enumerate(open(Root + 'label.txt', 'r', encoding='utf-8').read().splitlines())}
id2label = {i: val for i, val in enumerate(open(Root + 'label.txt', 'r', encoding='utf-8').read().splitlines())}

if 'uncased' in Model_name or 'bert-base-chinese' in Model_name:
    model = BertModel.from_pretrained(Model_name);
    Tokenizer = BertTokenizer.from_pretrained(Model_name)
elif 'skep' in Model_name:
    model = SkepModel.from_pretrained(Model_name);
    Tokenizer = SkepTokenizer.from_pretrained(Model_name)

Hidden_size = model.config['hidden_size']

Convert_func = convert_example_to_feature_normal if Convert == 'Normal' else convert_example_to_feature_special
Checkpoint = '_'.join([str(each) for each in ['cp',
                                              Root + Model_name + '/',
                                              Convert,
                                              Lstm,
                                              Res,
                                              Batch_func,
                                              Num_attention_heads,
                                              Loss_func
                                              ]])
Writer = LogWriter(logdir=Checkpoint)
os.mkdir(Checkpoint) if not os.path.exists(Checkpoint) else None
read_func = file_read_meituan if Root == '美团中文数据集/' else file_read_semeval

print(label2id)
print(Root, Num_sentis)
print(Checkpoint)
for each in config.items('Settings'):
    print('\t', each[0], '\t', each[1])
Train_loader, Dev_loader, Test_loader, Pred_loader = create_dataloader(Tokenizer, read_func, Convert_func, Root,
                                                                       Batch_size, ['train', 'dev', 'test', 'pred'])
Lr_scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=Learning_rate,
                                                warmup_steps=int(Num_epoch * 0.1 * len(Train_loader)), start_lr=0,
                                                end_lr=Learning_rate, verbose=False)
# 训练与测试
Model = Bert_LSTM_Attention(model,Hidden_size,len(label2id),Num_attention_heads,Num_sentis,Batch_func,Lstm,Res)
Optimizer = paddle.optimizer.AdamW(
    learning_rate=Lr_scheduler,
    parameters=Model.parameters(),
    weight_decay=0.01,
    apply_decay_param_fun=lambda x: x in [p.name for n, p in Model.named_parameters()if not any(nd in n for nd in ["bias", "norm"])])
train_bert_attention(Model, Train_loader, Num_epoch,  Writer, Lr_scheduler, Optimizer, Checkpoint, 20, Max_seq_len, Num_sentis, Loss_func)
