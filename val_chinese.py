from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
import numpy as np
import scipy.io as sio



# tra_results={}
# with open('../../../result/caption_train.txt') as f:
#         content = f.readlines()
#     # you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content] 
# # freq_wd = np.loadtxt('../../../result/caption_valid.txt')
# iter_freq = iter(content)
# results=[]
# for x in iter_freq:
#     x=x.rstrip().split(' ')
#     idx =int(x[0])
#     # print idx

#     x=np.array(x[1:-1])
    
#     sentence=' '.join(x)
#     # print sentence
#     # sentence = next(iter_freq)
#     if not idx in tra_results:
#         tra_results[idx] = []
#     tra_results[idx].append(sentence)
#     # print sentence
#     # print sentence
#     # sentence=' '.join()
#     # print sentence
# res_results={}  
# index=sio.loadmat('../../../val.mat')
# index=index['I'][0]


# for i in range(1000): 
#     if not i+8000+1 in res_results:
#         res_results[i+8000+1] = []
#     print index[i]-1
#     sentence = tra_results[index[i]]
#     # print sentence
#     res_results[i+8000+1]=sentence
    # print sentence[0]
res_results={}
with open('../../../out_valid_2.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
# freq_wd = np.loadtxt('../../../result/caption_valid.txt')
iter_freq = iter(content)
results=[]
for x in iter_freq:
    idx = x
    # print x
    sentence = next(iter_freq)
    results.append(sentence)
    # print sentence
    # print sentence
    # sentence=' '.join(sentence.rstrip().split(' '))
    # print sentence
    if not int(idx)-999 in res_results:
        res_results[int(idx)-999] = []
    res_results[int(idx)-999].append(sentence)
    # res_results.append({'image_id': idx, 'caption': sentence})
    
gt_results={}
with open('../../../result/caption_valid.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
# freq_wd = np.loadtxt('../../../result/caption_valid.txt')
iter_freq = iter(content)
idxold=8001
sentenceold=[]
sim=[]
for x in iter_freq:
    idx = x
    if(int(idx)>8088):
        idx=int(idx)-1
    # print x
    idx=int(idx)
    sentence = next(iter_freq)
    num = 0
    if not idx in gt_results:
        gt_results[idx] = []
    gt_results[idx].append(sentence)
    # for word in res_results[int(idx)]:
    #     if(sentence.find(word) >0):
    #         num += 1
    # sim.append(num)
    # # print num
    # sentenceold.append(sentence)
    # # print sentence
    # if(int(idx)!=idxold):
    #     idx_max=np.argmax(sim)
    #     idx_max=0
    #     # print sentenceold[idx_max]
    #     # print results[idxold-8001]
    #     sentence = sentenceold[idx_max]
    #     sentence=' '.join(sentence.rstrip().split(' '))
    #     print sentence
    #     gt_results[idxold]=sentence 
    #     # print sentence
    #     sentenceold=[]
    #     sim=[]

    #gt_results.append({'image_id': idx, 'caption': sentence})
    
    
#     idxold=int(idx)

# idx_max=np.argmax(sim)
# idx_max=0
# res_results
# # print sentenceold[idx_max]
# # print results[idxold-8001]
# sentence = sentenceold[idx_max]
# sentence=' '.join(sentence.rstrip().split(' '))
# gt_results[idxold]=sentence 
# sentenceold=[]
# sim=[]
gts = gt_results
res = res_results
# for imgId in range(1000):
#     gts[imgId] = gt_results[imgId]
#     res[imgId] = res_results[imgId]
# =================================================
# Set up scorers
# =================================================
# print 'tokenization...'
# tokenizer = PTBTokenizer()
# gts  = tokenizer.tokenize(gts)
# res = tokenizer.tokenize(res)
# =================================================
# Set up scorers
# =================================================
print 'setting up scorers...'
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    #(Meteor(),"METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr")  
]
# =================================================
# Compute scores
# =================================================
for scorer, method in scorers:
    print 'computing %s score...'%(scorer.method())
    score, scores = scorer.compute_score(gts, res)
    if type(method) == list:
        for sc, scs, m in zip(score, scores, method):
            #self.setEval(sc, m)
            #self.setImgToEvalImgs(scs, gts.keys(), m)
            print "%s: %0.3f"%(m, sc)
    else:
        #self.setEval(score, method)
        #self.setImgToEvalImgs(scores, gts.keys(), method)
        print "%s: %0.3f"%(method, score)