
from knets2.imports import *



def get_dataloaders(p):
    words,vocab,v2w=get_words_vocab_dict()
    p.v2w=v2w
    p.words=words
    p.vocab=vocab
    dataloaders={}
    for k in ('train','test'):
        print(k)
        dataloaders[k]=torch.utils.data.DataLoader(
            War_and_Peace_Dataset(
                words,
                vocab,
                p.extra_params[k]['start'],
                p.extra_params[k]['stop'],
                p.extra_params['max_seq_length'],
                p.batch_size,
                p.extra_params[k]['index_tracker'],
                p.device,
            ),
            batch_size=p.batch_size,
            shuffle=p.shuffle,
            num_workers=p.workers,
            persistent_workers=True,
        )     
    return dataloaders



class War_and_Peace_Dataset(torch.utils.data.Dataset):
    def __init__(
        _,
        words,
        vocab,
        start,
        stop,
        max_seq_length,
        batch_size,
        index_tracker,
        device,
    ):
        super(War_and_Peace_Dataset,_).__init__()
        packdict(_,locals())
        print("War_and_Peace::__init__",start,stop)
    def _identity(_,x):
        return x
    def __getitem__(_, index):
        src_data,tgt_data=get_src_tgt_data(
            _.words,
            _.vocab,
            _.start,
            _.stop,
            _.max_seq_length,
            _.batch_size,
            _.index_tracker,
        )
        src_data=src_data.to(_.device)
        tgt_data=tgt_data.to(_.device)
        return dict(src_data=src_data,tgt_data=tgt_data)
        #return src_data,tgt_data
    def __len__(_):
        return _.stop-_.start



def get_src_tgt_data(words,vocab,start,stop,max_seq_length,batch_size,index_tracker):
    v=vocab
    x=words
    #cm(len(x),type(x))

    src_data=[]
    tgt_data=[]
    for b in [0]:#range(batch_size):
        i=randint(start,stop-2*max_seq_length)
        if i not in index_tracker:
            index_tracker[i]=0
        #cb(i)
        #cb(i,x[i])
        index_tracker[i]+=1
        #print(len(x),type(x))
        #print(x[i:i+5])
        src=x[i:i+max_seq_length]
        tgt=x[i+max_seq_length:i+2*max_seq_length]
        for j in rlen(src):

            src[j]=v[src[j]]
            tgt[j]=v[tgt[j]]
        src_data=src#.append(src)
        tgt_data=tgt#.append(tgt)
    src_data=torch.from_numpy(na(src_data))
    tgt_data=torch.from_numpy(na(tgt_data))
    return src_data,tgt_data



def get_words_vocab_dict():
    x=file_to_text(opjD('War_and_Peace.txt')).lower() #(opjh('kllm/raw.txt')).lower()
    x=x.replace('- \n','')
    x=x.replace('-\n','')
    x=_strip(x)
    x=x.split(' ')
    x=remove_empty(x)
    print(' '.join(x))
    v={}
    index=1
    for w in x:
        if w not in v:
            v[w]=index
            index+=1
    v2w={}
    for k in v:
        w=v[k]
        v2w[w]=k
    src_vocab_size=len(v)
    tgt_vocab_size=src_vocab_size
    cm(src_vocab_size)
    words=x
    vocab=v
    return words,vocab,v2w



def _strip(text):
    return re.sub(r'[^a-zA-Z\']',' ',text)



if __name__ == '__main__':
    p=kws2class(
        batch_size=16,
        device='cpu',
        shuffle=True,
        workers=1,
        extra_params=dict(
            train=dict(start=0,stop=10000,index_tracker={},),
            test=dict(start=10000,stop=11000,index_tracker={},),
            
            max_seq_length=10,
        )
    )
    dl=get_dataloaders(p)
    from knets2.nets.building_a_transformer_with_py_torch import *
    max_seq_length = 10
    dropout = 0.1
    batch_size=16
    src_vocab_size=len(p.vocab)
    tgt_vocab_size=src_vocab_size
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.to(p.device)


    if True:
        
        ctr=0
        while True:
            for ii, datadic in enumerate(dl['train'], 0):
                for b in range(p.batch_size):
                    ws=[]
                    for i in range(p.extra_params['max_seq_length']):
                        q=datadic['src_data'][b,i]
                        j=int(q.detach().cpu().numpy())
                        ws.append(p.v2w[j])
                    cg(' '.join(ws))
                ctr+=1
                cb(ctr,r=0)


    if False:
        k='train'
        words,vocab=get_words_vocab_dict()
        wp=War_and_Peace_Dataset(
            words,
            vocab,
            p.extra_params[k]['start'],
            p.extra_params[k]['stop'],
            p.extra_params['max_seq_length'],
            p.batch_size,
            p.extra_params[k]['index_tracker'],
            p.device,
        )

#EOF
