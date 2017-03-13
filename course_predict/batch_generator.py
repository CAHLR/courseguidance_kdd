import numpy as np
import pickle
import json
from json2matrix import course_dict
from json2matrix import MAX_SEM, MAX_COURSE, COURSE_NUM, MAJOR_NUM, ENTRY_TYPE_NUM, GRADE_TYPE

def init_desc_dicts():
    bow_dir = '../preprocess/bow.ndarray'
    pca_dir = '../preprocess/pca.ndarray'
    key_dir = '../preprocess/desc_keys.json'
    bow = {}
    pca = {}
    keys = []
    with open(key_dir, 'r') as f:
        keys = json.load(f)
    bow_matrix = np.load(bow_dir)
    pca_matrix = np.load(pca_dir)
    counter = 0
    for key in keys:
        bow[key] = bow_matrix[counter, :]
        pca[key] = pca_matrix[counter, :]
        counter += 1
    return bow, pca, bow_matrix.shape[1]

BOW_dict, PCA_dict, bow_dim = init_desc_dicts()

def onehot_representation(matrix, output_dim=COURSE_NUM):
    '''
    input dim (sample_num, semester_num)
    output dim (sample_num, semester_num, COURSE_NUM)
    '''
    sample_num = matrix.shape[0]
    timestep = matrix.shape[1]
    label = np.zeros((sample_num, timestep, output_dim), dtype='float32')
    for i in range(sample_num):
        for j in range(timestep):
            one_hot = np.zeros(output_dim, dtype='float32')
            if matrix[i][j] > 0:
                one_hot[matrix[i][j] - 1] = 1
            label[i][j] = one_hot
    return label

def grade_multihot_representation(courses, grades):
    sample_num = courses.shape[0]
    semester_num = courses.shape[1]
    course_num = courses.shape[2]
    res_matrix = np.zeros((sample_num, semester_num, COURSE_NUM * GRADE_TYPE), dtype='int8')
    for i in range(sample_num):
        for j in range(semester_num):
            input_vec = np.zeros(COURSE_NUM * GRADE_TYPE, dtype='int8')
            for k in range(course_num):
                if courses[i, j, k] > 0:
                    grade_for_course = grades[i, j, k] - 1
                    input_vec[(courses[i,j,k]-1) * GRADE_TYPE + grade_for_course] = 1
            res_matrix[i][j] = input_vec
    return res_matrix

def multihot_representation(matrix, within_semester_dim=COURSE_NUM, data_type='float32'):
    '''
    input dim (sample_num, semester_num, course_num)
    output dim (sample_num, semester_num, COURSE_NUM)
    '''
    sample_num = matrix.shape[0]
    sem_num = matrix.shape[1]
    course_num = matrix.shape[2]
    res_matrix = np.zeros((sample_num, sem_num, within_semester_dim), dtype=data_type)
    for i in range(sample_num):
        for j in range(sem_num):
            input_vec = np.zeros(within_semester_dim, dtype=data_type)
            for k in range(course_num):
                if matrix[i][j][k] > 0.0:
                    input_vec[matrix[i][j][k] - 1] = 1.0
            res_matrix[i][j] = input_vec
    return res_matrix

def get_sample_matrix(y_list):
    weight_sample_list = []
    output_num = len(y_list)
    sample_num = y_list[0].shape[0]
    timestep_num = y_list[0].shape[1]
    label_dim = y_list[0].shape[2]
    zero_vec = np.zeros(label_dim, dtype='int8')
    label_matrix = np.zeros((output_num, sample_num, timestep_num, label_dim), dtype='int8')
    for i in range(output_num):
        label_matrix[i,:,:,:] = y_list[i]
        weight_sample_list.append(np.zeros((sample_num, timestep_num), dtype='float32'))
    for j in range(sample_num):
        for k in range(timestep_num):
            zero_label_num = 0
            for i in range(output_num):
                if (label_matrix[i, j, k, :] == zero_vec).all():
                    zero_label_num += 1
            if not zero_label_num == output_num:
                for i in range(output_num):
                    if not (label_matrix[i, j, k, :] == zero_vec).all():
                        weight_sample_list[i][j, k] = 1.0 / (output_num - zero_label_num)
    return weight_sample_list

def get_bow(course_matrix, dict_to_use='BOW'):
    '''
        in dim (batch_size, semester, courses)
    '''
    if dict_to_use == 'BOW':
        dict_to_use = BOW_dict
        feature_dim = bow_dim
    elif dict_to_use == 'PCA':
        dict_to_use = PCA_dict
        feature_dim = 128
    batch_size = course_matrix.shape[0]
    semester = course_matrix.shape[1]
    course_number= course_matrix.shape[2]
    bow_feature = np.zeros((batch_size, semester, feature_dim), dtype='float32')
    for i in range(batch_size):
        for j in range(semester):
            for k in range(course_number):
                if course_matrix[i, j, k] in dict_to_use:
                    # print(bow_feature.shape, dict_to_use[course_matrix[i, j, k]].shape)
                    bow_feature[i, j, :] += dict_to_use[course_matrix[i, j, k]]
    np.clip(bow_feature, 0., 1.)
    return bow_feature

def rnn_train_batch_generator(x_train, x_major, x_grad, x_gpa, y_train, use_pca=False, batch_size=1):
    sample_num = x_train.shape[0]
    desc_flag = 'PCA' if use_pca else 'BOW'
    for ndx in range(0, sample_num, batch_size):
        x_small = x_train[ndx:min(ndx+batch_size, sample_num), :, :]
        x_bow = get_bow(x_small, desc_flag)
        x_major_small = x_major[ndx:min(ndx+batch_size, sample_num), :, :]
        x_grad_small = x_grad[ndx:min(ndx+batch_size, sample_num), :]
        x_gpa_small = np.expand_dims(x_gpa[ndx:min(ndx+batch_size, sample_num), :], axis=2)
        y_small = y_train[ndx:min(ndx+batch_size, sample_num), :, :]
        y_bow = get_bow(y_small, desc_flag)

        x_major_multihot = multihot_representation(x_major_small, MAJOR_NUM, 'float32')
        x_grad_input = onehot_representation(x_grad_small, 2)
        train_input = multihot_representation(x_small, COURSE_NUM, 'int16')
        train_label = multihot_representation(y_small, COURSE_NUM, 'int16')
        yield train_input, x_major_multihot, x_grad_input, x_gpa_small, x_bow, y_bow, train_label

def rnn_evaluate_batch_generator(x_eval, x_eval_major, x_eval_grad, x_eval_gpa, y_eval, y_ppsk, use_pca=False, batch_size=1):
    sample_num = x_eval.shape[0]
    desc_flag = 'PCA' if use_pca else 'BOW'
    for ndx in range(0, sample_num, batch_size):
        x_small = x_eval[ndx:min(ndx+batch_size, sample_num), :, :]
        x_bow = get_bow(x_small, desc_flag)
        x_major_small = x_eval_major[ndx:min(ndx+batch_size, sample_num), :, :]
        x_eval_grad_small = x_eval_grad[ndx:min(ndx+batch_size, sample_num), :]
        x_gpa_small = np.expand_dims(x_eval_gpa[ndx:min(ndx+batch_size, sample_num), :], axis=2)
        y_small = y_eval[ndx:min(ndx+batch_size, sample_num), :]

        x_major_multihot = multihot_representation(x_major_small, MAJOR_NUM, 'float32')
        x_eval_grad_input = onehot_representation(x_eval_grad_small, 2)
        eval_input = multihot_representation(x_small, COURSE_NUM, 'int16')
        ppsk_small = y_ppsk[ndx:min(ndx+batch_size, sample_num)]
        yield eval_input, x_major_multihot, x_eval_grad_input, x_gpa_small, x_bow, y_small, ppsk_small

if __name__ == '__main__':
    print('Test this module: batch_generator')