# -*- coding: utf-8 -*-
import sys
import numpy as np
import json2matrix
import os
import time
import argparse
import json
import collections

import keras
from keras.layers import Input, Dense, LSTM, TimeDistributed, merge, Embedding, Masking, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.utils import np_utils
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam

from json2matrix import MAX_SEM, MAX_COURSE, COURSE_NUM, MAJOR_NUM, ENTRY_TYPE_NUM, GRADE_TYPE, MAX_MAJOR
from batch_generator import rnn_train_batch_generator, bow_dim
from batch_generator import multihot_representation
from evaluation import run_on_evalset, get_history_list, load_my_model
from custom_loss import course_loss, TOP1, BPR

# model path
model_dir = '../models'
config_dir = ''
# -----------------------This part is for the LSTM model---------------------------

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def load_hyper_parameters(json_dir):
    my_dict = {}
    with open(json_dir, 'r') as data_file:
        my_dict = json.load(data_file)
    return my_dict

# multihot-input multihot-output
def build_the_model(hyper_parameter_dict, word2vec, has_major, has_grad, has_gpa, semester_number):
    '''
        return an untrained model
    '''
    input_matrix = Input(shape=(semester_number, COURSE_NUM), name='input course multihot')

    input_major = Input(shape=(semester_number, MAJOR_NUM), name='input major multihot')

    input_grad = Input(shape=(semester_number, 2), name='input grad status')

    input_gpa = Input(shape=(semester_number, 1), name='gpa')

    masked_input = Masking(mask_value=0., name='mask zero')(temporal_input)

    lstm_out = LSTM(hyper_parameter_dict['lstm_dim'], 
                    return_sequences=True, 
                    name='lstm layer out', 
                    dropout_W=hyper_parameter_dict['dropout_rate'])(masked_input)

    concat_list = [lstm_out]
    input_list = [input_matrix]
    if has_major:
        concat_list.append(input_major)
        input_list.append(input_major)
    if has_grad:
        concat_list.append(input_grad)
        input_list.append(input_grad)
    if has_gpa:
        concat_list.append(input_gpa)
        input_list.append(input_gpa)

    if has_major or has_grad or has_gpa or has_bow:
        concat_vector = merge(concat_list, mode='concat')
    else:
        concat_vector = lstm_out

    output_layer = TimeDistributed(Dense(COURSE_NUM, activation='softmax'), name='softmax_out')(concat_vector)

    model = Model(input=input_list, output=output_layer)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.summary()
    print('Neural network has been initialized.')
    return model

# base
def train_rnn_model(model_save_name, hyper_parameter_dict, word2vec, x_train, x_major, x_grad, x_gpa, y_train, 
    has_major=False, has_grad=False, has_gpa=False, has_bow=False, use_pca=False, detail_result=False, my_validation_data=None):
    '''
        train a rnn model
    '''
    print('Train set dimension', x_train.shape, y_train.shape)

    model = build_the_model(hyper_parameter_dict, word2vec, has_major, has_grad, has_gpa, x_train.shape[1])

    # start training
    start_time = time.time()
    print('Going to train for', hyper_parameter_dict['epoch_num'], 'epochs, with mini-batch size', hyper_parameter_dict['my_batch_size'])
    for e in range(hyper_parameter_dict['epoch_num']):
        shuffled_rank = np.random.permutation(x_train.shape[0])
        x_train = x_train[shuffled_rank, :, :]
        x_major = x_major[shuffled_rank, :, :]
        x_grad = x_grad[shuffled_rank, :]
        x_gpa = x_gpa[shuffled_rank, :]
        y_train = y_train[shuffled_rank, :, :]      
        print("---------------------------------------\nEpoch %d\n---------------------------------------" % (e+1))
        for train_input, x_major_multihot, x_grad_input, x_gpa_small, x_bow, y_bow, train_label in rnn_train_batch_generator(x_train, x_major, x_grad, x_gpa, y_train, use_pca, hyper_parameter_dict['gen_data_batch_size']):
            my_input = [train_input]
            if has_major:
                my_input.append(x_major_multihot)
            if has_grad:
                my_input.append(x_grad_input)
            if has_gpa:
                my_input.append(x_gpa_small)
            if has_bow:
                my_input.append(x_bow)
            model.fit(my_input, train_label, batch_size=hyper_parameter_dict['my_batch_size'], nb_epoch=1, validation_data=my_validation_data)
    end_time = time.time()
    
    # saving the model
    model_save_dir = os.path.join(model_dir, model_save_name)
    with open(os.path.join(model_dir, model_save_name+'.json'), 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_save_dir)
    print('Model saved in', model_save_dir, 'Training process is done in', int((end_time - start_time) / 60), 'minutes.')

# -----------------------------LSTM part ends here---------------------------------

def get_validation_data(x, x_major, y, size=1):
    val_size = min(size, x.shape[0])
    indices = np.random.choice(x.shape[0], val_size, replace=False)
    new_x = np.delete(x, indices, axis=0)
    new_major = np.delete(x_major, indices, axis=0)
    new_y = np.delete(y, indices, axis=0)
    valid_x = multihot_representation(x[indices, :, :])
    valid_major = multihot_representation(x_major[indices, :, :], MAJOR_NUM)
    valid_y = multihot_representation(y[indices, :, :])
    return new_x, new_major, new_y, ([valid_x, valid_major], valid_y)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sem', help='Student who has more semester will be ignored.', 
                        type=int, default=12)
    parser.add_argument('--course', help='Output node number.', 
                        type=int, default=6)
    parser.add_argument('--train', help='Specify the name of the model you want to save to.', type=str)
    parser.add_argument('--test', help='Specify the name of the model you want to load and test.', type=str)
    parser.add_argument('--desc', help='Description of the model structure.', type=str, default='')

    args = parser.parse_args(arguments)
    hyper_parameter_dict = load_hyper_parameters(config_dir)

    semester_num = args.sem
    course_num = args.course
    description = args.desc

    print('Preparing the dataset.')
    
    course_matrix, major_matrix, grad_matrix, gpa_matrix, target_matrix, eval_ppsk = json2matrix.get_rnn_dataset(hyper_parameter_dict['eval_semester'], 
                                                                                                    semester_num, MAX_COURSE, MAX_COURSE)
    # validation
    # x_train, x_major, y_train, validation_set = get_validation_data(x_train, x_major, y_train, 4096)
    if args.train != None:
        # Training part
        embedding_weights = None
        print('Training multi-hot-input multi-output model.')
        model_save_name = args.train

        train_rnn_model(model_save_name, hyper_parameter_dict, embedding_weights, 
                        course_matrix[0], major_matrix[0], grad_matrix[0], gpa_matrix[0], target_matrix[0], 
                        has_major=hyper_parameter_dict['has_major'], has_grad=hyper_parameter_dict['has_grad'], 
                        has_gpa=hyper_parameter_dict['has_gpa'], has_bow=hyper_parameter_dict['has_bow'], 
                        use_pca=hyper_parameter_dict['use_pca'], my_validation_data=None)
        # saving the model description
        description_name = model_save_name + '.desc'
        description_dir = os.path.join(model_dir, description_name)

    if args.test != None:
        print('Evaluating multi-hot-input multi-output model.')
        recall, mrr, detail_result = run_on_evalset(hyper_parameter_dict['eval_semester'], args.test, course_matrix[1], 
            major_matrix[1], grad_matrix[1], gpa_matrix[1], target_matrix[1], eval_ppsk, 
            has_major=hyper_parameter_dict['has_major'], has_grad=hyper_parameter_dict['has_grad'], 
            has_gpa=hyper_parameter_dict['has_gpa'], has_bow=hyper_parameter_dict['has_bow'], use_pca=hyper_parameter_dict['use_pca'], detail_output=False)
        print(detail_result)
        print('Done! Recall@10:', recall, 'MRR@10', mrr)
    return 0

if __name__ == '__main__':
    device_arg = sys.argv[1]
    config_dir = os.path.join(sys.path[0], 'config.json')
    print('Using config file:', config_dir)
    available_device = ['cpu', 'gpu', 'gpu0', 'gpu1', 'gpu2', 'gpu3']
    if device_arg not in available_device:
        print("You must specify a valid device to run the code.")
        sys.exit(-1)
    sys.exit(main(sys.argv[2:]))