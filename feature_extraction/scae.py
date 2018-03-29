import numpy as np
from feature_extraction.cae import CAE
from feature_extraction.mcae import MCAE
import tensorflow as tf

class SCAE():
    """Stacked Convolutional Autoencoder with single- and multi-loss support.

    """
    def __init__(self, nb_bands=None, nb_cae=None, weight_decay=0.0,
                 ckpt_path='checkpoints', patience=10, num_epochs=1000,
                 pre_trained=True, loss_mode='mcae'):
        """
        Args:
            nb_bands (int): Number of bands (dimensionality).
            nb_cae (int): Number of CAEs.
            weight_decay (float): (Default: 1e-4)
            gpu_list (list of int, None): List of GPU ids that the model is trained on.
                If None, the model is trained on all available GPUs.
            model_checkpoint_path (str): Filepath to save the model file after every epoch.
            loss_mode (str): Specifies whether the CAE is single- or multi-loss.
                Supported values are `single` and `multi`.
                (Default: `multi`)
            weights (str): Filepath to the weights file to load. (Default: None)
            loss (str): Loss function used for MCAE.
                Supported functions are `mse` and `mae`.

        """

        self.N = 256
        self.nb_cae = nb_cae
        self.ckpt_path = ckpt_path
        loss_weights=[1.0, 1e-1, 1e-2, 1e-2]

        if nb_bands is not None:
            self.nb_bands = np.append(nb_bands ,
                               np.ones(nb_cae-1,dtype=np.int32)*self.N)

        self.weight_decay= weight_decay
        self.patience = patience
        self.loss_weights = loss_weights
        self.num_epochs = num_epochs
        self.loss_mode = loss_mode
        
    def fit(self, X_train, X_val, batch_size=128):
        """ Trains the model on the given dataset for 500 epochs.

        Args:
            X_train (arr): Training data.
            X_val (arr): Validation data.
            batch_size (int): Number of samples per gradient update.

        """
        for i in range(self.nb_cae):
            print('Training CAE #%d...' % i)
                
            if self.loss_mode == 'cae':
            
                cae = CAE(self.nb_bands[i], 
                          ckpt_path=self.ckpt_path+'/cae%d' % i,
                              loss_weights=self.loss_weights, 
                              weight_decay=self.weight_decay,
                              patience=self.patience, 
                              num_epochs=self.num_epochs)
            elif self.loss_mode == 'mcae':
                cae = MCAE(self.nb_bands[i], 
                          ckpt_path=self.ckpt_path+'/mcae%d' % i,
                              loss_weights=self.loss_weights, 
                              weight_decay=self.weight_decay,
                              patience=self.patience, 
                              num_epochs=self.num_epochs) 
            else:
                raise ValueError('Not a valid loss_mode.')
            
            cae.fit(X_train, X_val, batch_size=batch_size)
            if i < self.nb_cae-1:
                X_train = cae.transform(X_train)
                X_val = cae.transform(X_val)

        return 1


    def transform(self, X):
        """Generates output reconstructions for the input samples.

        Args:
            X (arr): The input data.

        Returns:
            Reconstruction of input data.

        """
        output = np.zeros((X.shape[0], X.shape[1], self.N*self.nb_cae),
                           dtype=np.float32)
        X = X[np.newaxis]
        for i in range(self.nb_cae):
            
            with tf.Graph().as_default():
                with tf.Session() as sess:            
                    ckpt = tf.train.get_checkpoint_state(self.ckpt_path+'/cae%d/' % i)  
                    if ckpt and ckpt.model_checkpoint_path:
                        # if checkpoint exists, restore the parameters and set epoch_n and i_iter
                        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta',
                                                           clear_devices=True)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        inputs = tf.get_collection('inputs')[0]
                        training = tf.get_collection('training')[0]
                        hidden_output = tf.get_collection('hidden_output')[0]
                    else:
                        raise FileNotFoundError('Cannot Find Checkpoint')
                            

                    X = sess.run(hidden_output, feed_dict={inputs: X,
                                                training: False})
                    output[:,:, self.N*i:self.N*(i+1)] = X[0]
            

        return output
