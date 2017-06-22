# -*- coding: utf-8 -*-
import jieba

def caption2words(filename, filename_suffix, wds, freq, idx_offset):
    seq_no = -1
    fw = open('caption_' + filename_suffix + '.txt', 'w')
    fw_v = open('caption_vec' + filename_suffix + '.txt', 'w')

    with open(filename) as f:
        for line in f:
            if len(line) >= 10:
                fw.write(str(seq_no + idx_offset)+' ')
                fw_v.write(str(seq_no + idx_offset)+' ')
                seg_list = jieba.cut(line, cut_all=False)
                for wd in seg_list:
                    if wd != '\n':
                        wd = wd.encode('utf-8')
                        fw.write(wd + ' ')
                        if wd in wds:
                            idx = wds.index(wd)
                            freq[idx] = freq[idx] + 1
                            tmp = str(idx)
                            fw_v.write(tmp + ' ')
                        else:
                            tmp = str(len(wds))
                            fw_v.write(tmp + ' ')
                            wds.append(wd)
                            freq.append(1)
                    else:
                        fw.write('\n')
                        fw_v.write('\n')
            else:	# if this line is sample_No
                seq_no = seq_no + 1
    f.close()
    fw_v.close()
    fw.close()
    print '样本数：' + str(seq_no+1) + '　个\n'

def caption2letter(filename, filename_suffix, wds, freq, idx_offset):
    seq_no = -1
    fw = open('caption_' + filename_suffix + '.txt', 'w') #　储存依字符分割后的结果
    fw_v = open('caption_vec' + filename_suffix + '.txt', 'w') # 储存依字符分割后的数值化向量结果
    with open(filename) as f:
        for line in f:
            if len(line) >= 10: 
                fw.write(str(seq_no + idx_offset)+' ')
                fw_v.write(str(seq_no + idx_offset)+' ')
                line = unicode(line, "utf-8")
                for wd in line:
                    if wd != '\n':
                        wd = wd.encode('utf-8')
                        fw.write(wd + ' ')
                        if wd in wds:
                            idx = wds.index(wd)
                            freq[idx] = freq[idx] + 1
                            tmp = str(idx)
                            fw_v.write(tmp + ' ')
                        else:
                            tmp = str(len(wds))
                            fw_v.write(tmp + ' ')
                            wds.append(wd)
                            freq.append(1)
                    else:
                        fw.write('\n')
                        fw_v.write('\n')
            else:	# if this line is sample_No
                seq_no = seq_no + 1
    f.close()
    fw_v.close()
    fw.close()
    print '样本数：' + str(seq_no+1) + '　个\n'


#######################################
# 分词
#######################################
filename_suffix = "valid"  # Here choose the data set: { "train", "test", ...}

wds = list()
freq = list()

# print('【jieba】正在分词……\n')
# caption2words(filename_suffix + '_test.txt', filename_suffix, wds, freq, 1)

print('正在拆分字符……\n')
# train: idx_offset = 1; valid: idx_offset = 8001
caption2letter(filename_suffix + '.txt', filename_suffix, wds, freq, 8001)

f = open('wds_freq_' + filename_suffix + '.txt', 'w')
count = len(wds)
print '词/字：' + str(count) + ' 个\n'
for i in range(0,count):
    f.write(wds[i] + '　' + str(freq[i]) + '\n')
f.close()

#######################################
# 后期处理
#######################################
print('==> start...')
filename = 'test_final.txt'
fw = open('test_final_out.txt', 'w')
s_num = 10 # sentence number of each picture
with open(filename) as f:
    seq_no = 0
    mark = False
    for line in f:
        line = line.split('.\n')
        line = line[0].split(' ')
        line_tmp = list()
        for wd in line:
            if wd not in line_tmp:
                line_tmp.append(wd)
        if line_tmp[-1] in ['的', '在', '里', '着', '和', '有']:
            del line_tmp[-1]
        if line_tmp[-1] in ['的', '在', '里', '着', '和', '有']:
            del line_tmp[-1]
        if seq_no % 10 == 0: # 如果为第１个
            line_hold = line_tmp
            if len(line_hold) <= 7:
                mark = True # 需要优化
        elif mark == True:
            if len(line_tmp) > len(line_hold):
                line_hold = line_tmp
        if seq_no % 10 == 9:
            fw.write(str(9000 + seq_no / 10) + ' ' + ''.join(line_hold[1:]))
            fw.write('\n')
            mark = False
        seq_no = seq_no + 1
fw.close()
print("==> over...")
