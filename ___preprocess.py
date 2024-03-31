
from utilz2 import *

def _strip(text):
	return re.sub(r'[^a-zA-Z\']',' ',text)

def get_x_v_dict():
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
	return dict=(x=x,v=v,)
	



def get_src_tgt_data(vocab,words,start,stop,max_seq_length,batch_size,index_tracker):
	v,x=vocab,words
	src_data=[]
	tgt_data=[]
	for b in range(batch_size):
		i=randint(start,stop-2*max_seq_length)
		if i not in index_tracker:
			index_tracker[i]=0
		index_tracker[i]+=1
		src=x[i:i+max_seq_length]
		tgt=x[i+max_seq_length:i+2*max_seq_length]
		#cm(src)
		#cr(tgt)
		for j in rlen(src):

			src[j]=v[src[j]]
			tgt[j]=v[tgt[j]]
		#cb(src)
		#cg(tgt)
		src_data.append(src)
		tgt_data.append(tgt)
	src_data=torch.from_numpy(na(src_data))
	tgt_data=torch.from_numpy(na(tgt_data))
	return src_data,tgt_data

