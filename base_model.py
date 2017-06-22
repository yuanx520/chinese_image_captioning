import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm


from dataset import *
from utils.words import *
from utils.coco.coco import *
from utils.coco.pycocoevalcap.eval import *

class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True 
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)

    def load_img(self, img_file):    
        """ Load and preprocess an image. """  
        img = cv2.imread(img_file)

        if self.bgr:
            temp = img.swapaxes(0, 2)
            temp = temp[::-1]
            img = temp.swapaxes(0, 2)
        img = cv2.resize(img, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        img = img[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1], :]
        img = img - self.mean
        return img

    def load_imgs(self, img_files):
        """ Load and preprocess a list of images. """
        imgs = []
        for img_file in img_files:
            imgs.append(self.load_img(img_file))
        imgs = np.array(imgs, np.float32)
        return imgs

class BaseModel(object):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.batch_size = params.batch_size if mode=='train' else 1

        self.cnn_model = params.cnn_model
        self.train_cnn = params.train_cnn
        self.init_lstm_with_fc_feats = params.init_lstm_with_fc_feats if self.cnn_model=='vgg16' else False

        self.class_balancing_factor = params.class_balancing_factor

        self.save_dir = os.path.join(params.save_dir, self.cnn_model+"/")

        self.word_table = HzTable(params.vocab_size, params.dim_embed, params.max_sent_len, params.word_table_file)
	#self.word_table.build()
        #self.word_table.load()
	self.word_table.build('/home/rll/Documents/PR_Project/image_captioning/result_single/wds_freq_train.txt') 

        self.img_loader = ImageLoader(params.mean_file)
        self.img_shape = [224, 224, 3]

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.build()
        self.saver = tf.train.Saver(max_to_keep = 100)
  


    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train, contexts=None, feats=None):
        raise NotImplementedError()

   
    def train_chinese(self, sess, train_data):
        """ Train the model. """
        print("Training the model...")
        params = self.params
        num_epochs = params.num_epochs
         
        for epoch_no in tqdm(list(range(num_epochs)), desc='epoch'):
            for idx in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()

                               
                    # Train RNN only 
                feats_indx, _, _ = batch
                fc7_feats = train_data.fc7_feats_all[feats_indx]
                conv5_3_feats = train_data.conv5_3_feats[feats_indx]
                #print conv5_3_feats.shape
                conv5_3_feats = conv5_3_feats.reshape([self.batch_size,49,512])
                #conv5_3_feats = tf.reshape(conv5_3_feats, [self.batch_size,49,512])
                #print conv5_3_feats.shape
                if self.init_lstm_with_fc_feats:
                    contexts, feats = sess.run([self.conv_feats, self.fc_feats], feed_dict={self.fc_feats:fc7_feats, self.conv_feats:conv5_3_feats, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=True, contexts=contexts, feats=feats)
                else:
                    contexts = sess.run(self.conv_feats, feed_dict={self.fc_feats:fc7_feats, self.conv_feats:conv5_3_feats, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=True, contexts=contexts)

                _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict)
                print(" Loss0=%f Loss1=%f" %(loss0, loss1))

                if (global_step + 1) % params.save_period == 0:
                    self.save(sess)
                #sentence = self.word_table.indices_to_sent(result.squeeze())
                #print sentence

            train_data.reset()

        self.save(sess)

        print("Training complete.")

    

    def val_chinese(self, sess, val_data):
        """ Validate the model. """
        print("Validating the model ...")
        results = []
        result_dir = self.params.val_result_dir

        # Generate the captions for the images
        for k in tqdm(list(range(val_data.count))):
	    batch = val_data.next_batch()
	    #print batch
            feats_indx= batch
            fc7_feats = val_data.fc7_feats_all[feats_indx]
            conv5_3_feats = val_data.conv5_3_feats[feats_indx]
            conv5_3_feats = conv5_3_feats.reshape([self.batch_size,49,512])
            
            if self.init_lstm_with_fc_feats:
                contexts, feats = sess.run([self.conv_feats, self.fc_feats], feed_dict={self.fc7_feats:fc7_feats, self.conv5_3_feats:conv5_3_feats, self.is_train:False})
                feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts, feats=feats)
            else:
                contexts = sess.run(self.conv_feats, feed_dict={self.fc_feats:fc7_feats, self.conv_feats:conv5_3_feats, self.is_train:False})
                feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts)

            result = sess.run(self.results, feed_dict=feed_dict)
	    print result.squeeze()
            sentence = self.word_table.indices_to_sent(result.squeeze())
            results.append({'image_id': val_data.img_ids[k], 'caption': sentence})
	    print sentence
	#np.savetxt('./baseline_result',result)

            # Save the result in an image file
            #img = mpimg.imread(img_file)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.title(sentence)
            #plt.savefig(os.path.join(result_dir, img_name+'_result.jpg'))

        val_data.reset() 
        scorer = COCOEvalCap(val_coco, val_res_coco)
        scorer.evaluate()
        # Evaluate these captions
        #val_res_coco = val_coco.loadRes2(results)
        #scorer = COCOEvalCap(val_coco, val_res_coco)
        #scorer.evaluate()
        print("Validation complete.")

    def test_chinese(self, sess, test_data, show_result=False):
        """ Test the model. """
        print("Testing the model ...")
        result_file = self.params.test_result_file
        result_dir = self.params.test_result_dir
        captions = []
        files=open('test.txt','w')
        # Generate the captions for the images
        i=9000;
        for k in tqdm(list(range(test_data.count))):
            batch = test_data.next_batch()
	    #print batch
            feats_indx= batch
            fc7_feats = test_data.fc7_feats_all[feats_indx]
            conv5_3_feats = test_data.conv5_3_feats[feats_indx]
            conv5_3_feats = conv5_3_feats.reshape([self.batch_size,49,512])
            if self.train_cnn:
                feed_dict = self.get_feed_dict(batch, is_train=False)
            else:
        
                if self.init_lstm_with_fc_feats:
                    contexts, feats = sess.run([self.conv_feats, self.fc_feats], feed_dict={self.fc_feats:fc7_feats, self.conv_feats:conv5_3_feats, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts, feats=feats)
                else:
                    contexts = sess.run(self.conv_feats, feed_dict={self.fc_feats:fc7_feats, self.conv_feats:conv5_3_feats, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts)

            result = sess.run(self.results, feed_dict=feed_dict)
            
            sentence = self.word_table.indices_to_sent(result.squeeze())
            captions.append(sentence)
            print sentence
            
            files.write(str(i))
            files.write(' ')
            i=i+1
            """
            a=[]
            for i in range(len(sentence)-1):
                if(sentence[i]==' '| sentence[i]=='.'):
                    a.append(sentence[i])
                a.append(sentence[i])
                a.append(sentence[i+1])
                a.append(' ')
            print(a)
            """
            files.write(sentence)
            files.write('\n')
            # Save the result in an image file
            
        files.close()
        # Save the captions to a file
        results = pd.DataFrame({'caption':captions})
        results.to_csv(result_file)
        print("Testing complete.")
  
    def save(self, sess):
        """ Save the model. """
        print(("Saving model to %s" % self.save_dir))
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        """ Load the model. """
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        print(checkpoint)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

    def load2(self, data_path, session, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading CNN model from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        miss_count = 0
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                       #print("Variable %s:%s loaded" %(op_name, param_name))
                    except ValueError:
                        miss_count += 1
                       #print("Variable %s:%s missed" %(op_name, param_name))
                        if not ignore_missing:
                            raise
        print("%d variables loaded. %d variables missed." %(count, miss_count))


