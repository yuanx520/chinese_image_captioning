import os
import math
import numpy as np
import pandas as pd
import cPickle as pickle
import h5py

from utils.words import *
from utils.coco.coco import *

class DataSet_chinese():
	def __init__(self, img_ids, img_fc1, img_fc2, img_conv, caps=None, masks=None, batch_size=1, is_train=False, shuffle=False):
		self.img_ids = np.array(img_ids)
		self.fc6_feats_all = np.array(img_fc1)
		self.fc7_feats_all = np.array(img_fc2)
		self.conv5_3_feats = np.array(img_conv)
		self.caps = np.array(caps)
		self.masks = np.array(masks)
		self.batch_size = batch_size
		self.is_train = is_train
		self.shuffle = shuffle
		self.setup()

	def setup(self):
		""" Setup the dataset. """
		self.count = len(self.img_ids)
		self.num_batches = int(self.count * 1.0 / self.batch_size)
		self.current_index = 0
		self.indices = list(range(self.count))
	#print self.indices
		self.reset()

	def reset(self):
		""" Reset the dataset. """
		self.current_index = 0
		if self.shuffle:
			np.random.shuffle(self.indices)

	def next_batch(self):
		""" Fetch the next batch. """
		assert self.has_next_batch()
		start, end = self.current_index, self.current_index + self.batch_size
		current_idx = self.indices[start:end]
		if self.is_train:
			caps = self.caps[current_idx]
			masks = self.masks[current_idx]
			self.current_index += self.batch_size

			return current_idx, caps, masks
		else:
			self.current_index += self.batch_size
			return current_idx

	def has_next_batch(self):
		""" Determine whether there is any batch left. """
		return self.current_index + self.batch_size <= self.count

def prepare_train_data(args):
	""" Prepare relevant data for training the model. """
	image_dir, caption_file, annotation_file = args.train_image_dir, args.train_caption_file, args.train_annotation_file
	init_embed_with_glove, vocab_size, word_table_file, glove_dir = args.init_embed_with_glove, args.vocab_size, args.word_table_file, args.glove_dir
	dim_embed, batch_size, max_sent_len = args.dim_embed, args.batch_size, args.max_sent_len

	coco = COCO(caption_file)
	coco.filter_by_cap_len(max_sent_len)

	print("Building the word table...")
	word_table = WordTable(vocab_size, dim_embed, max_sent_len, word_table_file)
	if not os.path.exists(word_table_file):
		if init_embed_with_glove:
			word_table.load_glove(glove_dir)
		word_table.build(coco.all_captions())
		word_table.save()
	else:
		word_table.load()
	print("Word table built. Number of words = %d" %(word_table.num_words))

	coco.filter_by_words(word_table.all_words())
	
	if not os.path.exists(annotation_file):
		annotations = process_captions(coco, image_dir, annotation_file)
	else:
		annotations = pd.read_csv(annotation_file)

	img_ids = annotations['image_id'].values
	img_files = annotations['image_file'].values
	captions = annotations['caption'].values
	print("Number of training captions = %d" %(len(captions)))

	caps, masks = symbolize_captions(captions, word_table)

	print("Building the training dataset...")
	dataset = DataSet(img_ids, img_files, caps, masks, batch_size, True, True)
	print("Dataset built.")
	return coco, dataset

def prepare_val_data(args):
	""" Prepare relevant data for validating the model. """
	image_dir, caption_file = args.val_image_dir, args.val_caption_file

	coco = COCO(caption_file)

	img_ids = list(coco.imgs.keys())
	img_files = [os.path.join(image_dir, coco.imgs[img_id]['file_name']) for img_id in img_ids]
  
	print("Building the validation dataset...")
	dataset = DataSet(img_ids, img_files)
	print("Dataset built.")
	return coco, dataset


def prepare_test_data(args):
	""" Prepare relevant data for testing the model. """
	image_dir = args.test_image_dir

	files = os.listdir(image_dir)
	img_files = [os.path.join(image_dir, f) for f in files if f.lower().endswith('.jpg')]
	img_ids = list(range(len(img_files)))

	print("Building the testing dataset...")    
	dataset = DataSet(img_ids, img_files)
	print("Dataset built.")
	return dataset


def process_captions(coco, image_dir, annotation_file):
	""" Build an annotation file containing the training information. """
	captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
	image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
	image_files = [os.path.join(image_dir, coco.imgs[img_id]['file_name']) for img_id in image_ids]
	annotations = pd.DataFrame({'image_id': image_ids, 'image_file': image_files, 'caption': captions})
	annotations.to_csv(annotation_file)
	return annotations


def symbolize_captions(captions, word_table):
	""" Translate the captions into the indicies of their words in the vocabulary, and get their masks. """
	caps = []
	masks = []
	for cap in captions:
		idx, mask = word_table.symbolize_sent(cap)
		caps.append(idx)
		masks.append(mask)
	return np.array(caps), np.array(masks)

def symbolize_chinese_captions(caption_file, hz_table):
	""" Translate the captions into the indicies of their words in the vocabulary, and get their masks. """
	with open(caption_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 
	#print len(content)
	img_idx = []
	img_caption=[]
	img_cap_mask=[]
	for x in content:
	    x = x.split()
	    x = map(int, x )
            x.append(-1)
	    img_idx.append(x[0])
	    caption, mask = hz_table.symbolize_sent(x[1:])
	    img_caption.append(caption)
 	    img_cap_mask.append(mask)
	#print img_cap_mask[0]
	return img_idx, img_caption ,img_cap_mask
def feats_expand(features, idx):
	feat_expand=[];
	for i in idx:
		feat_expand.append(features[i-1])
	return feat_expand
def feats_expand_test(features, num):
	feat_expand=[];
	for x in features:
		feat_expand.append(x)
		for i in range(num):
			feat_expand.append(x+np.random.normal(0,0.6,x.shape))
	return feat_expand
def prepare_chinese_train_data(args):
	""" Prepare relevant data for training the model. """
	image_dir, caption_file, annotation_file = args.train_image_dir, args.train_caption_file, args.train_annotation_file
	init_embed_with_glove, vocab_size, word_table_file, glove_dir = args.init_embed_with_glove, args.vocab_size, args.word_table_file, args.glove_dir
	dim_embed, batch_size, max_sent_len = args.dim_embed, args.batch_size, args.max_sent_len

	#coco = COCO(caption_file)
	#coco.filter_by_cap_len(max_sent_len)

	print("Building the word table...")
	hz_table = HzTable(vocab_size, dim_embed, max_sent_len, word_table_file)
	# hz_table.build('/home/rll/Documents/PR_Project/image_captioning/jieba/wds_freq_train.txt')
	# [img_idx, img_caption ,img_cap_mask] = symbolize_chinese_captions('/home/rll/Documents/PR_Project/image_captioning/jieba/caption_vec_train.txt', hz_table)
	hz_table.build('/home/rll/Documents/PR_Project/image_captioning/result_single/wds_freq_train.txt')
	[img_idx, img_caption ,img_cap_mask] = symbolize_chinese_captions('/home/rll/Documents/PR_Project/image_captioning/result_single/caption_vec_train.txt', hz_table)
	vgg_f1=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_fc1_feature_78660340.h5','r')
	vgg_f2=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_fc2_feature_677004464.h5','r')
	vgg_conv=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_block5_pool_feature_661509428.h5','r')
	vgg_f1_group_key = vgg_f1.keys()
	vgg_f1_test = list(vgg_f1[vgg_f1_group_key[0]])
	vgg_f1_train = list(vgg_f1[vgg_f1_group_key[1]])
	vgg_f1_val = list(vgg_f1[vgg_f1_group_key[2]])
	vgg_f2_group_key = vgg_f2.keys()
	vgg_f2_test = list(vgg_f2[vgg_f2_group_key[0]])
	vgg_f2_train=list(vgg_f2[vgg_f2_group_key[1]])
	vgg_f2_val=list(vgg_f2[vgg_f2_group_key[2]])
	vgg_conv_group_key = vgg_conv.keys()
	vgg_conv_test=list(vgg_conv[vgg_conv_group_key[0]])
	vgg_conv_train=list(vgg_conv[vgg_conv_group_key[1]])
	vgg_conv_val=list(vgg_conv[vgg_conv_group_key[2]])
	vgg_f1_train_expand = feats_expand (vgg_f1_train, img_idx) 
	vgg_f2_train_expand = feats_expand (vgg_f2_train, img_idx) 
	vgg_conv_train_expand = feats_expand (vgg_conv_train, img_idx) 
	print("Building the training dataset...")
	dataset = DataSet_chinese(img_idx, vgg_f1_train_expand, vgg_f2_train_expand, vgg_conv_train_expand, img_caption, img_cap_mask, batch_size, True, True)
	print("Dataset built.")
	return dataset

def prepare_chinese_val_data(args):
	""" Prepare relevant data for validating the model. """
        image_dir, caption_file, annotation_file = args.train_image_dir, args.train_caption_file, args.train_annotation_file
	init_embed_with_glove, vocab_size, word_table_file, glove_dir = args.init_embed_with_glove, args.vocab_size, args.word_table_file, args.glove_dir
	dim_embed, batch_size, max_sent_len = args.dim_embed, args.batch_size, args.max_sent_len

	image_dir, caption_file = args.val_image_dir, args.val_caption_file
	[img_idx, img_caption ,img_cap_mask] = symbolize_chinese_captions('/home/rll/Documents/PR_Project/image_captioning/result_single/caption_vec_valid.txt', hz_table)
	#coco = COCO(caption_file)
	print img_idx
	vgg_f1=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_fc1_feature_78660340.h5','r')
	vgg_f2=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_fc2_feature_677004464.h5','r')
	vgg_conv=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_block5_pool_feature_661509428.h5','r')
	vgg_f1_group_key = vgg_f1.keys()
	vgg_f1_test = list(vgg_f1[vgg_f1_group_key[0]])
	vgg_f1_train = list(vgg_f1[vgg_f1_group_key[1]])
	vgg_f1_val = list(vgg_f1[vgg_f1_group_key[2]])
	vgg_f2_group_key = vgg_f2.keys()
	vgg_f2_test = list(vgg_f2[vgg_f2_group_key[0]])
	vgg_f2_train=list(vgg_f2[vgg_f2_group_key[1]])
	vgg_f2_val=list(vgg_f2[vgg_f2_group_key[2]])
	vgg_conv_group_key = vgg_conv.keys()
	vgg_conv_test=list(vgg_conv[vgg_conv_group_key[0]])
	vgg_conv_train=list(vgg_conv[vgg_conv_group_key[1]])
	vgg_conv_val=list(vgg_conv[vgg_conv_group_key[2]])
	vgg_f1_val_expand = feats_expand_test (vgg_f1_val, 0) 
	vgg_f2_val_expand = feats_expand_test (vgg_f2_val, 0) 
	vgg_conv_val_expand = feats_expand_test (vgg_conv_val, 0) 
	print("Building the validation dataset...")
	img_idx = range(len(vgg_f1_val_expand))
	#print img_idx
	dataset = DataSet_chinese(img_idx, vgg_f1_val_expand, vgg_f2_val_expand, vgg_conv_val_expand, img_caption, img_cap_mask) 
	print("Dataset built.")
	return dataset
def prepare_chinese_test_data(args):
	""" Prepare relevant data for validating the model. """
	image_dir, caption_file = args.val_image_dir, args.val_caption_file

	#coco = COCO(caption_file)
	vgg_f1=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_fc1_feature_78660340.h5','r')
	vgg_f2=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_fc2_feature_677004464.h5','r')
	vgg_conv=h5py.File('/home/rll/Documents/PR_Project/image_captioning/image_vgg19_block5_pool_feature_661509428.h5','r')
	vgg_f1_group_key = vgg_f1.keys()
	vgg_f1_test = list(vgg_f1[vgg_f1_group_key[0]])
	vgg_f1_train = list(vgg_f1[vgg_f1_group_key[1]])
	vgg_f1_val = list(vgg_f1[vgg_f1_group_key[2]])
	vgg_f2_group_key = vgg_f2.keys()
	vgg_f2_test = list(vgg_f2[vgg_f2_group_key[0]])
	vgg_f2_train=list(vgg_f2[vgg_f2_group_key[1]])
	vgg_f2_val=list(vgg_f2[vgg_f2_group_key[2]])
	vgg_conv_group_key = vgg_conv.keys()
	vgg_conv_test=list(vgg_conv[vgg_conv_group_key[0]])
	vgg_conv_train=list(vgg_conv[vgg_conv_group_key[1]])
	vgg_conv_val=list(vgg_conv[vgg_conv_group_key[2]])
	vgg_f1_val_expand = feats_expand_test (vgg_f1_val, 0) 
	vgg_f2_val_expand = feats_expand_test (vgg_f2_val, 0) 
	vgg_conv_val_expand = feats_expand_test (vgg_conv_val, 0) 
	print("Building the validation dataset...")
	img_idx = range(len(vgg_f1_val_expand))
	#print img_idx
	dataset = DataSet_chinese(img_idx, vgg_f1_val_expand, vgg_f2_val_expand, vgg_conv_val_expand) 
	print("Dataset built.")
	return dataset

