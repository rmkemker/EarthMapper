import tensorflow as tf
import math
import os
import numpy as np
import tf_initializers

class EMA(object):
    def __init__(self, decay=0.99):
        self.avg = None
        self.decay = decay
        self.history = []
    def update(self, X):
        if not self.avg:
            self.avg = X
        else:
            self.avg = self.avg * self.decay + (1-self.decay) * X
        self.history.append(self.avg)

class SSMLP(object):

    def __init__(self, layer_sizes, unsupervised_weights=None, num_epochs=150, 
                 batch_size=32, weight_decay=None, 
                 starter_learning_rate=2e-3, ckpt_path='checkpoints',
                 patience=50,
                 activation='pelu', unsupervised_cost_fn='mse',
                 weight_initializer='glorot_normal',
                 mu = 0.25, eval_batch_size=512):
        
        '''
        Attributes
        '''
        self.L = len(layer_sizes) - 1  # number of layers
        self.layer_sizes = layer_sizes
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.ckpt_path = ckpt_path
        self.starter_learning_rate = starter_learning_rate
        self.patience = patience
        self.lr_patience = int(patience/2)
        self.activation = activation
        self.unsup_cost_fn = unsupervised_cost_fn
        self.weight_initializer = tf_initializers.get(weight_initializer)
        self.mu = mu
        self.eval_batch_size = eval_batch_size
                
        if unsupervised_weights is None:
            self.unsup_weights = np.ones(len(layer_sizes)-2)
        else:
            self.unsup_weights = unsupervised_weights
   
        tf.reset_default_graph()
        
        inputs = tf.placeholder(tf.float32, shape=(None,self.layer_sizes[0]),name='inputs')
        outputs = tf.placeholder(tf.int32, name='outputs')
        class_weights = tf.placeholder(tf.float32, name='class_weights')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        training = tf.placeholder(tf.bool,name='training')

        shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))  # shapes of linear layers

        weights = {'W': [self.weight_initializer(s, "W") for s in shapes],
                   'Wb':[self.bi(1.0, self.layer_sizes[l+1], "Wb") for l in range(self.L)],
                   'V': [self.weight_initializer(s[::-1], "V") for s in shapes],
                   'Vb':[self.bi(1.0, self.layer_sizes[l], "Vb") for l in range(self.L)],
                   'a1':[tf.Variable(tf.ones(1)) for l in range(self.L-1)],
                   'b1':[tf.Variable(tf.ones(1)) for l in range(self.L-1)],
                   'a2':[tf.Variable(tf.ones(1)) for l in range(self.L-1)],
                   'b2':[tf.Variable(tf.ones(1)) for l in range(self.L-1)]}
                    
        tf.add_to_collection("inputs", inputs)
        tf.add_to_collection("outputs", outputs)
        tf.add_to_collection("training", training)
        tf.add_to_collection('learning_rate', learning_rate)
        tf.add_to_collection('class_weights', class_weights)
        
        eps = tf.constant(1e-10)
        
        #Encoder
        h = inputs
        encoder_layers = [inputs]
        for i in range(self.L-1):
            h = tf.matmul(h, weights['W'][i])
            h = tf.add(h, weights['Wb'][i])             
            h = tf.layers.batch_normalization(h, training=training)
            
            if i == self.L - 2:
                encoded_path = self.act(h, weights["a1"][i], weights["b1"][i])
                h = tf.matmul(encoded_path, weights['W'][self.L-1])  
                h = tf.add(h, weights['Wb'][self.L-1])
                y = tf.nn.softmax(h) 
            else:
                h = self.act(h, weights["a1"][i], weights["b1"][i])
                encoder_layers.append(h)
        
        tf.add_to_collection("y", y)
        
        #Decoder

        h = encoded_path
        decoder_layers = []
        for i in range(self.L-2, -1, -1):
            h = tf.matmul(h, weights['V'][i])
            h = tf.add(h, weights['Vb'][i]) 
            
            h = tf.layers.batch_normalization(h, training=training)

            if i == 0:
                decoded_path = h
                decoder_layers.append(decoded_path)
            else:
                h = self.act(h, weights["a2"][i], weights["b2"][i])
                decoder_layers.append(h)
            h = tf.add(h, encoder_layers[i])

        
        decoder_layers = decoder_layers[::-1]
        u_cost = 0.0
        if self.unsup_cost_fn == 'mae':
            for i in range(len(encoder_layers)):
                u_cost += (self.unsup_weights[i] * tf.reduce_mean(tf.abs(encoder_layers[i] - decoder_layers[i]),1))
        else:
            for i in range(len(encoder_layers)):
                u_cost += (self.unsup_weights[i] * tf.reduce_mean(tf.square(encoder_layers[i] - decoder_layers[i]),1))
        
#        tf.add_to_collection("u_cost", u_cost)

#        clipped_output = tf.clip_by_value(outputs, 1e-10, 1.0)
#        cost = -tf.reduce_mean(tf.reduce_sum(outputs * tf.multiply(tf.log(y+eps),class_weights), 1))  # supervised cost
#        loss = cost + u_cost   # total cost

#        tf.add_to_collection('cost', cost)

#        if self.weight_decay is not None:
#            weight_decay = tf.nn.l2_loss(weights['W'][0]) + tf.nn.l2_loss(weights['V'][0])
#            for i in range(1,self.L):
#                weight_decay = weight_decay + tf.nn.l2_loss(weights['W'][i]) + tf.nn.l2_loss(weights['V'][i])
#
#            loss = loss + self.weight_decay * weight_decay # total cost

#        if self.activation == 'pelu':
#            loss = loss - 1000.0 * tf.minimum( tf.reduce_min(weights['a1']), 0)
#            loss = loss - 1000.0 * tf.minimum( tf.reduce_min(weights['b1']), 0) 
#            loss = loss - 1000.0 * tf.minimum( tf.reduce_min(weights['a2']), 0)
#            loss = loss - 1000.0 * tf.minimum( tf.reduce_min(weights['b2']), 0) 

#        tf.add_to_collection('loss', loss)

        one_hot = tf.cast(tf.one_hot(outputs, layer_sizes[-1]) , tf.float32)
        loss_per = -tf.reduce_sum(one_hot * tf.multiply(tf.log(y+eps),class_weights), 1) + u_cost# supervised cost
        loss = tf.reduce_mean(loss_per)
        
        reg_loss = 0.0
        if self.weight_decay is not None:
            for i in range(0,self.L):
                reg_loss = reg_loss + self.weight_decay*tf.nn.l2_loss(weights['W'][i]) 
                reg_loss = reg_loss + self.weight_decay*tf.nn.l2_loss(weights['V'][i])
                
        pelu_loss = 0.0
        if self.activation == 'pelu':
            pelu_loss = pelu_loss - 1000.0 * tf.minimum( tf.reduce_min(weights['a1']), 0)
            pelu_loss = pelu_loss - 1000.0 * tf.minimum( tf.reduce_min(weights['b1']), 0) 
            pelu_loss = pelu_loss - 1000.0 * tf.minimum( tf.reduce_min(weights['a2']), 0)
            pelu_loss = pelu_loss - 1000.0 * tf.minimum( tf.reduce_min(weights['b2']), 0) 

        loss = loss + reg_loss + pelu_loss
        loss_per = loss_per + reg_loss + pelu_loss
        tf.add_to_collection('loss_per_t', loss_per)
        tf.add_to_collection('loss_t', loss)


        pred = tf.argmax(y, 1)
        tf.add_to_collection('pred', pred)
        truth = tf.argmax(outputs, 1)
        correct_prediction = tf.equal(tf.argmax(y, 1), truth)  # no of correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
        tf.add_to_collection('accuracy', accuracy)
        
        opt = tf.contrib.opt.NadamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = opt.minimize(loss)        
        
        tf.add_to_collection('train_step', train_step)
                
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, self.ckpt_path+'/model.ckpt', 0)

    def bi(self, inits, size, name):
        return tf.Variable(inits * tf.ones([size]), name=name)
    
    def wi(self, shape, name):
        return tf.Variable(tf.random_normal(shape, name=name)/ math.sqrt(shape[0])) 

    def act(self, z, a, b):
        """Applies PELU activation function.

        Args:
            z (tensor): data to apply activation function on.
            a (float): `a` parameter.
            b (float): `b` parameter.
        """ 
        if self.activation == 'relu':
            return tf.nn.relu(z)
        elif self.activation == 'tanh':
            return tf.nn.tanh(z)
        elif self.activation == 'leakyrelu':
            return tf.nn.relu(z) - 0.1 * tf.nn.relu(-z)
        elif self.activation == 'pelu':
            positive = tf.nn.relu(z) * a / (b + 1e-9)
            negative = a * (tf.exp((-tf.nn.relu(-z)) / (b + 1e-9)) - 1)
            return negative + positive
        elif self.activation == 'elu':
            return tf.nn.elu(z)
        else:
            raise NotImplementedError('Only RELU, ELU, TANH, and PELU activations are supported')
    
    def fit(self, X_train, y_train, X_val, y_val):
        
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
              
        with tf.Session() as sess:            
            print("=== Training RonNet===")
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path+'/')  # get latest checkpoint (if any)
            if ckpt and ckpt.model_checkpoint_path:
                # if checkpoint exists, restore the parameters and set epoch_n and i_iter
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta',
                                                   clear_devices=True)
                saver.restore(sess, ckpt.model_checkpoint_path)
                inputs = tf.get_collection('inputs')[0]
                outputs = tf.get_collection('outputs')[0]
                training = tf.get_collection('training')[0]
                class_weights = tf.get_collection('class_weights')[0]
                train_step = tf.get_collection('train_step')[0]
                loss_t = tf.get_collection('loss_t')[0]
                loss_per_t = tf.get_collection('loss_per_t')[0]
                learning_rate = tf.get_collection('learning_rate')[0]
            else:
                raise FileNotFoundError('Cannot Find Checkpoint')
            
            #Compute initial val_loss
            best_loss = 1000000.0            

            
            lr = self.starter_learning_rate
                        
            pc = 0 # patience counter 
            lrc = 0 # learning rate patience counter
            
            num_classes= np.max(y_train)+1  
            if self.mu is None:
                cw = np.ones(num_classes)
            else:
                cw = np.bincount(y_train, minlength=num_classes)       
                cw = np.sum(cw)/cw
                cw[np.isinf(cw)] = 0
                cw = self.mu * np.log(cw)
                cw[cw < 1] = 1
            
            t_msg1 = '\rEpoch %d/%d (%d/%d) -- lr=%1.0e  -- train=%1.2f'
            v_msg1 = ' -- val=%1.2f'
            val_loss_arr = np.zeros((X_val.shape[0]), np.float32)
            ema = EMA()
            val_samples = X_val.shape[0]
            idx = np.arange(X_train.shape[0])
            for e in range(self.num_epochs):
                
                np.random.shuffle(idx)  
                
                
                for i in range(0, X_train.shape[0], self.batch_size):
                    _, loss = sess.run([train_step, loss_t],
                                            feed_dict={inputs: X_train[idx[i:i+self.batch_size]],
                                                       outputs: y_train[idx[i:i+self.batch_size]],
                                                       training: True,
                                                       learning_rate: lr,
                                                       class_weights:cw})
                    ema.update(loss)
                    print(t_msg1 % (e+1,self.num_epochs,i+1,X_train.shape[0],lr, ema.avg), end="")
                
                #Calculate validation loss/acc
                eval_mb = 256
                for i in range(0,  val_samples, eval_mb):       
                    start = np.min([i, val_samples - eval_mb])
                    end = np.min([i + eval_mb, val_samples])
                    val_loss_arr[start:end] = sess.run(loss_per_t,
                          feed_dict={inputs: X_val[start:end],
                                     outputs: y_val[start:end],
                                     training: False,
                                     class_weights:cw})
                val_loss = np.mean(val_loss_arr)
                
                
                print(v_msg1 % (val_loss), end="")
    
                if val_loss < best_loss and not math.isnan(val_loss):
                    saver.save(sess, self.ckpt_path+'/model.ckpt',e)
                    best_loss = val_loss
                    print(' -- best_val_loss=%1.2f' % best_loss)
                    pc = 0
                    lrc=0
                else:
                    pc += 1
                    lrc += 1
                    if pc > self.patience:
                        break
                    elif lrc >= self.lr_patience:
                        lrc = 0
                        lr /= 10.0                        
                    print(' -- Patience=%d/%d' % (pc, self.patience))
                
                if math.isnan(val_loss):
                    break

    def predict(self, X):

        n = X.shape[0]
        
        with tf.Graph().as_default():
        
            with tf.Session() as sess:
                
                ckpt = tf.train.get_checkpoint_state(self.ckpt_path+'/')  # get latest checkpoint (if any)
                if ckpt and ckpt.model_checkpoint_path:
                    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
                    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta',
                                                       clear_devices=True)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    inputs = tf.get_collection('inputs')[0]
                    outputs = tf.get_collection('outputs')[0]
                    training = tf.get_collection('training')[0]
                    pred = tf.get_collection('pred')[0]
                else:
                    raise FileNotFoundError('No checkpoint loaded')
                
                prediction = np.zeros(n, dtype=np.int32)
        
                for i in range(0, n, self.batch_size):
                    start = np.min([i, n - self.batch_size])
                    end = np.min([i + self.batch_size, n])
        
                    X_test = X[start:end]
                    y_test = X[start:end]
        
                    prediction[start:end] = sess.run(pred,
                              feed_dict={inputs: X_test,
                                         outputs: y_test,
                                         training: False})

        return prediction

    def predict_proba(self, X):

        n = X.shape[0]
        
        with tf.Graph().as_default():
        
            with tf.Session() as sess:        
                ckpt = tf.train.get_checkpoint_state(self.ckpt_path+'/')  # get latest checkpoint (if any)
                if ckpt and ckpt.model_checkpoint_path:
                    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
                    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta',
                                                       clear_devices=True)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    inputs = tf.get_collection('inputs')[0]
                    training = tf.get_collection('training')[0]
                    proba = tf.get_collection('y')[0]
                else:
                    raise FileNotFoundError('No checkpoint loaded')
                
                prediction = np.zeros((n, self.layer_sizes[-1]), dtype=np.float32)
        
                for i in range(0, n, self.batch_size):
                    start = np.min([i, n - self.batch_size])
                    end = np.min([i + self.batch_size, n])
        
                    X_test = X[start:end]
        
                    prediction[start:end] = sess.run(proba,
                              feed_dict={inputs: X_test,
                                         training: False})

        return prediction
    
    

