import math
import os
import tensorflow as tf
import numpy as np
from base_model import *
from utils.nn import *

class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN part...")
        """
        if self.cnn_model=='vgg16':
            self.build_vgg16()

        elif self.cnn_model=='resnet50':
            self.build_resnet50()

        elif self.cnn_model=='resnet101':
            self.build_resnet101()

        else:
            self.build_resnet152()
        """
        self.build_chinese()
        print("CNN part built.")

   
    def basic_block(self, input_feats, name1, name2, is_train, bn, c, s=2):
        """ A basic block of ResNets. """
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, bn, None)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, bn, c):
        """ Another basic block of ResNets. """
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_chinese(self):
    	bn = self.params.batch_norm
        fc7_feats = tf.placeholder(tf.float32, [self.batch_size]+[4096])
        conv5_3_feats = tf.placeholder(tf.float32, [self.batch_size]+[49,512])
        is_train = tf.placeholder(tf.bool)
        """
        fc6_feats = fully_connected(feats_im, 4096, 'fc6')
        fc6_feats = nonlinear(fc6_feats, 'relu')
        if self.train_cnn:
            fc6_feats = dropout(fc6_feats, 0.5, is_train)

        fc7_feats = fully_connected(fc6_feats, 4096, 'fc7')
        """
        # pool5_feats_flat = tf.reshape(conv5_3_feats, [self.batch_size, -1]) 

        #pool5_feats_flat = tf.reshape(conv5_3_feats, [self.batch_size, 49,512]) 

        # pool5_feats_flat.set_shape([self.batch_size, 49*512])
        conv5_3_feats_flat = tf.reshape(conv5_3_feats, [self.batch_size, 49, 512])
        self.conv_feats = conv5_3_feats_flat
        self.conv_feat_shape = [49, 512]

        self.fc_feats = fc7_feats
        self.fc_feat_shape = [4096]

        #self.imgs = imgs
        self.is_train = is_train


    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN part...")

        params = self.params
        bn = params.batch_norm      

        batch_size = self.batch_size                        
        num_ctx = self.conv_feat_shape[0]                   
        dim_ctx = self.conv_feat_shape[1]                   

        num_words = self.word_table.num_words
        max_sent_len = params.max_sent_len
        num_lstm = params.num_lstm
        dim_embed = params.dim_embed
        dim_hidden = params.dim_hidden
        dim_dec = params.dim_dec

        if not self.train_cnn:
            contexts = tf.placeholder(tf.float32, [batch_size] + self.conv_feat_shape)
            if self.init_lstm_with_fc_feats:
                feats = tf.placeholder(tf.float32, [batch_size] + self.fc_feat_shape)
        else:
            contexts = self.conv_feats
            if self.init_lstm_with_fc_feats:
                feats = self.fc_feats

        sentences = tf.placeholder(tf.int32, [batch_size, max_sent_len])
        masks = tf.placeholder(tf.float32, [batch_size, max_sent_len])        

        is_train = self.is_train

        self.word_weight = np.exp(-np.array(self.word_table.word_freq)*self.class_balancing_factor)

        self.position_weight = np.exp(-np.array(list(range(max_sent_len)))*0.003)

        # initialize the word embedding
        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] for i in range(num_words)])
        if params.fix_embed_weight:
            emb_w = tf.convert_to_tensor(idx2vec, tf.float32)
        else:
            emb_w = weight('emb_w', [num_words, dim_embed], init_val=idx2vec, group_id=1)

        # initialize the decoding layer
        dec_w = weight('dec_w', [dim_dec, num_words], group_id=1)  
        if params.init_dec_bias: 
            dec_b = bias('dec_b', [num_words], init_val=self.word_table.word_freq)
        else:
            dec_b = bias('dec_b', [num_words], init_val=0.0)
 
        # compute the mean context
        context_mean = tf.reduce_mean(contexts, 1)
       
        # initialize the LSTMs
        lstm = tf.contrib.rnn.LSTMCell(dim_hidden, initializer=tf.random_normal_initializer(stddev=0.03)) 

        if self.init_lstm_with_fc_feats:
            init_feats = feats
        else:
            init_feats = context_mean

        if num_lstm == 1:
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc1'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn1'+str(i), is_train, bn, 'tanh')
            memory = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc2'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn2'+str(i), is_train, bn, 'tanh')
            output = tf.identity(temp)

            state = tf.contrib.rnn.LSTMStateTuple(memory, output)                   

        else:
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc11'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn11'+str(i), is_train, bn, 'tanh')
            memory1 = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc12'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn12'+str(i), is_train, bn, 'tanh')
            output1 = tf.identity(temp)

            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc21'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn21'+str(i), is_train, bn, 'tanh')
            memory2 = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc22'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn22'+str(i), is_train, bn, 'tanh')
            output = tf.identity(temp)

            state1 = tf.contrib.rnn.LSTMStateTuple(memory1, output1)                
            state2 = tf.contrib.rnn.LSTMStateTuple(memory2, output)                 

        loss0 = 0.0
        results = []
        scores = []
        context_flat = tf.reshape(contexts, [-1, dim_ctx])  
       
        # Generate the words one by one 
        for idx in range(max_sent_len):

            # Attention mechanism
            context_encode1 = fully_connected(context_flat, dim_ctx, 'att_fc11', group_id=1) 
            context_encode1 = batch_norm(context_encode1, 'att_bn11', is_train, bn, None) 

            context_encode2 = fully_connected_no_bias(output, dim_ctx, 'att_fc12', group_id=1) 
            context_encode2 = batch_norm(context_encode2, 'att_bn12', is_train, bn, None) 
            context_encode2 = tf.tile(tf.expand_dims(context_encode2, 1), [1, num_ctx, 1])                 
            context_encode2 = tf.reshape(context_encode2, [-1, dim_ctx])    

            context_encode = context_encode1 + context_encode2  
            context_encode = nonlinear(context_encode, 'relu')  
            context_encode = dropout(context_encode, 0.5, is_train)

            alpha = fully_connected(context_encode, 1, 'att_fc2', group_id=1)                 
            alpha = batch_norm(alpha, 'att_bn2', is_train, bn, None)
            alpha = tf.reshape(alpha, [-1, num_ctx])                                                           
            alpha = tf.nn.softmax(alpha)                                                                       
         
            if idx == 0:   
                word_emb = tf.zeros([batch_size, dim_embed])
                weighted_context = tf.identity(context_mean)
            else:
                word_emb = tf.cond(is_train, lambda: tf.nn.embedding_lookup(emb_w, sentences[:, idx-1]), lambda: word_emb)
                weighted_context = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), 1)
            
            # Apply the LSTMs 
            print num_lstm
            if num_lstm == 1:
                with tf.variable_scope("lstm"):
                    output, state = lstm(tf.concat([weighted_context, word_emb],1), state)
            else:
                with tf.variable_scope("lstm1"):
                    output1, state1 = lstm(weighted_context, state1)

                with tf.variable_scope("lstm2"):
                    output, state2 = lstm(tf.concat([word_emb, output1],1), state2)
            
            # Compute the logits
            expanded_output = tf.concat([output, weighted_context, word_emb],1)

            logits1 = fully_connected(expanded_output, dim_dec, 'dec_fc', group_id=1)
            logits1 = nonlinear(logits1, 'tanh')
            logits1 = dropout(logits1, 0.5, is_train)

            logits2 = tf.nn.xw_plus_b(logits1, dec_w, dec_b)

            # Update the loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=sentences[:, idx])
            cross_entropy = cross_entropy * masks[:, idx]
            loss0 += tf.reduce_sum(cross_entropy)
 
            # Update the result
            max_prob_word = tf.argmax(logits2, 1)
            results.append(max_prob_word)

            probs = tf.nn.softmax(logits2)
            score = tf.reduce_max(probs, 1)
            scores.append(score)

            # Prepare for the next iteration
            word_emb = tf.cond(is_train, lambda: word_emb, lambda: tf.nn.embedding_lookup(emb_w, max_prob_word))          
            tf.get_variable_scope().reuse_variables()                           

        # Get the final result
        results = tf.stack(results, axis=1)
        scores = tf.stack(scores, axis=1)

        # Compute the final loss 
        loss0 = loss0 / tf.reduce_sum(masks)
        if self.train_cnn:
            loss1 = params.weight_decay * (tf.add_n(tf.get_collection('l2_0')) + tf.add_n(tf.get_collection('l2_1')))
        else:
            loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_1'))
        loss = loss0 + loss1
        
        # Build the solver
        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        tvars = tf.trainable_variables()
        gs, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 3.0)
        opt_op = solver.apply_gradients(zip(gs, tvars), global_step=self.global_step)

        self.contexts = contexts
        if self.init_lstm_with_fc_feats:
            self.feats = feats
        self.sentences = sentences
        self.masks = masks

        self.loss = loss
        self.loss0 = loss0
        self.loss1 = loss1
        self.opt_op = opt_op

        self.results = results
        self.scores = scores
        
        print("RNN part built.")   
    def build_rnn_2(self):
        """ Build the RNN. """
        print("Building the RNN part...")

        params = self.params
        bn = params.batch_norm      

        batch_size = self.batch_size                        
        num_ctx = self.conv_feat_shape[0]                   
        dim_ctx = self.conv_feat_shape[1]                   

        num_words = self.word_table.num_words
        max_sent_len = params.max_sent_len
        num_lstm = params.num_lstm
        dim_embed = params.dim_embed
        dim_hidden = params.dim_hidden
        dim_dec = params.dim_dec

        if not self.train_cnn:
            contexts = tf.placeholder(tf.float32, [batch_size] + self.conv_feat_shape)
            if self.init_lstm_with_fc_feats:
                feats = tf.placeholder(tf.float32, [batch_size] + self.fc_feat_shape)
        else:
            contexts = self.conv_feats
            if self.init_lstm_with_fc_feats:
                feats = self.fc_feats

        sentences = tf.placeholder(tf.int32, [batch_size, max_sent_len])
        masks = tf.placeholder(tf.float32, [batch_size, max_sent_len])        

        is_train = self.is_train

        self.word_weight = np.exp(-np.array(self.word_table.word_freq)*self.class_balancing_factor)

        self.position_weight = np.exp(-np.array(list(range(max_sent_len)))*0.003)

        # initialize the word embedding
        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] for i in range(num_words)])
        if params.fix_embed_weight:
            emb_w = tf.convert_to_tensor(idx2vec, tf.float32)
        else:
            emb_w = weight('emb_w', [num_words, dim_embed], init_val=idx2vec, group_id=1)

        # initialize the decoding layer
        dec_w = weight('dec_w', [dim_dec, num_words], group_id=1)  
        if params.init_dec_bias: 
            dec_b = bias('dec_b', [num_words], init_val=self.word_table.word_freq)
        else:
            dec_b = bias('dec_b', [num_words], init_val=0.0)
 
        # compute the mean context
        context_mean = tf.reduce_mean(contexts, 1)
       
        # initialize the LSTMs
        lstm = tf.contrib.rnn.LSTMCell(dim_hidden, initializer=tf.random_normal_initializer(stddev=0.03)) 

        if self.init_lstm_with_fc_feats:
            init_feats = feats
        else:
            init_feats = context_mean

        if num_lstm == 1:
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc1'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn1'+str(i), is_train, bn, 'tanh')
            memory = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc2'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn2'+str(i), is_train, bn, 'tanh')
            output = tf.identity(temp)

            state = tf.contrib.rnn.LSTMStateTuple(memory, output)                   

        else:
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc11'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn11'+str(i), is_train, bn, 'tanh')
            memory1 = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc12'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn12'+str(i), is_train, bn, 'tanh')
            output1 = tf.identity(temp)

            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc21'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn21'+str(i), is_train, bn, 'tanh')
            memory2 = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc22'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn22'+str(i), is_train, bn, 'tanh')
            output = tf.identity(temp)

            state1 = tf.contrib.rnn.LSTMStateTuple(memory1, output1)                
            state2 = tf.contrib.rnn.LSTMStateTuple(memory2, output)                 

        loss0 = 0.0
        results = []
        scores = []
        context_flat = tf.reshape(contexts, [-1, dim_ctx])  
       
        # Generate the words one by one 
        for idx in range(max_sent_len):

            # Attention mechanism
            context_encode1 = fully_connected(context_flat, dim_ctx, 'att_fc11', group_id=1) 
            context_encode1 = batch_norm(context_encode1, 'att_bn11', is_train, bn, None) 

            context_encode2 = fully_connected_no_bias(output, dim_ctx, 'att_fc12', group_id=1) 
            context_encode2 = batch_norm(context_encode2, 'att_bn12', is_train, bn, None) 
            context_encode2 = tf.tile(tf.expand_dims(context_encode2, 1), [1, num_ctx, 1])                 
            context_encode2 = tf.reshape(context_encode2, [-1, dim_ctx])    

            context_encode = context_encode1 + context_encode2  
            context_encode = nonlinear(context_encode, 'relu')  
            context_encode = dropout(context_encode, 0.5, is_train)

            alpha = fully_connected(context_encode, 1, 'att_fc2', group_id=1)                 
            alpha = batch_norm(alpha, 'att_bn2', is_train, bn, None)
            alpha = tf.reshape(alpha, [-1, num_ctx])                                                           
            alpha = tf.nn.softmax(alpha)                                                                       
         
            if idx == 0:   
                word_emb = tf.zeros([batch_size, dim_embed])
                weighted_context = tf.identity(context_mean)
            else:
                word_emb = tf.cond(is_train, lambda: tf.nn.embedding_lookup(emb_w, sentences[:, idx-1]), lambda: word_emb)
                weighted_context = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), 1)
            
            # Apply the LSTMs 
            if num_lstm == 1:
                with tf.variable_scope("lstm"):
                    output, state = lstm(tf.concat([weighted_context, word_emb],1), state)
            else:
                with tf.variable_scope("lstm1"):
                    output1, state1 = lstm(weighted_context, state1)

                with tf.variable_scope("lstm2"):
                    output, state2 = lstm(tf.concat([word_emb, output1],1), state2)
            
            # Compute the logits
            expanded_output = tf.concat([output, weighted_context, word_emb],1)

            logits1 = fully_connected(expanded_output, dim_dec, 'dec_fc', group_id=1)
            logits1 = nonlinear(logits1, 'tanh')
            logits1 = dropout(logits1, 0.5, is_train)

            logits2 = tf.nn.xw_plus_b(logits1, dec_w, dec_b)

            # Update the loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=sentences[:, idx])
            cross_entropy = cross_entropy * masks[:, idx]
            loss0 += tf.reduce_sum(cross_entropy)
 
            # Update the result
            max_prob_word = tf.argmax(logits2, 1)
            results.append(max_prob_word)

            probs = tf.nn.softmax(logits2)
            score = tf.reduce_max(probs, 1)
            scores.append(score)

            # Prepare for the next iteration
            word_emb = tf.cond(is_train, lambda: word_emb, lambda: tf.nn.embedding_lookup(emb_w, max_prob_word))          
            tf.get_variable_scope().reuse_variables()                           

        # Get the final result
        results = tf.stack(results, axis=1)
        scores = tf.stack(scores, axis=1)

        # Compute the final loss 
        loss0 = loss0 / tf.reduce_sum(masks)
        if self.train_cnn:
            loss1 = params.weight_decay * (tf.add_n(tf.get_collection('l2_0')) + tf.add_n(tf.get_collection('l2_1')))
        else:
            loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_1'))
        loss = loss0 + loss1
        
        # Build the solver
        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        tvars = tf.trainable_variables()
        gs, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 3.0)
        opt_op = solver.apply_gradients(zip(gs, tvars), global_step=self.global_step)

        self.contexts = contexts
        if self.init_lstm_with_fc_feats:
            self.feats = feats
        self.sentences = sentences
        self.masks = masks

        self.loss = loss
        self.loss0 = loss0
        self.loss1 = loss1
        self.opt_op = opt_op

        self.results = results
        self.scores = scores
        
        print("RNN part built.")        

    def get_feed_dict(self, batch, is_train, contexts=None, feats=None):
        """ Get the feed dictionary for the current batch. """
        if is_train:
            # training phase
            img_feat, sentences, masks = batch
            #imgs = self.img_loader.load_imgs(img_files)
            sentences_new=np.zeros(sentences.shape);
            for i in range(self.batch_size):
                sentences_now=map(lambda x: x if x<11557 else 11557,sentences[i, :])
                sentences_new[i,:]=sentences_now
                word_weight = self.word_weight[sentences_now]                
                masks[i, :] = masks[i, :] * word_weight
                masks[i, :] = masks[i, :] * self.position_weight

            if self.train_cnn:
                return {self.imgs: imgs, self.sentences: sentences, self.masks: masks, self.is_train: is_train}
            else:
                if self.init_lstm_with_fc_feats:
                    return {self.contexts: contexts, self.feats: feats, self.sentences: sentences, self.masks: masks, self.is_train: is_train}        
                else:
                    return {self.contexts: contexts, self.sentences: sentences_new, self.masks: masks, self.is_train: is_train} 

        else:
            # testing or validation phase
            #img_files = batch 
            img_feat = batch 
            #imgs = self.img_loader.load_imgs(img_files)
            fake_sentences = np.zeros((self.batch_size, self.params.max_sent_len), np.int32)

            if self.train_cnn:
                return {self.imgs: imgs, self.sentences: fake_sentences, self.is_train: is_train}
            else:
                if self.init_lstm_with_fc_feats:
                    return {self.contexts: contexts, self.feats: feats, self.sentences: fake_sentences, self.is_train: is_train}        
                else:
                    return {self.contexts: contexts, self.sentences: fake_sentences, self.is_train: is_train} 


