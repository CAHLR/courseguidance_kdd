import sys
import csv
import numpy as np
import os
import time
import argparse
import json
import collections

import keras
from keras.models import model_from_json, Model
from json2matrix import MAX_SEM, MAX_COURSE, COURSE_NUM, MAJOR_NUM 
from json2matrix import get_rnn_dataset
from json2matrix import course_dict, major_dict
from json2matrix import student_detail_dict, course_detail_dict, college_dict
from batch_generator import rnn_evaluate_batch_generator

CSV_HEADER = ['ppsk', 'student type', 'student college', 'student major',
            'course number', 'label course name', 'label department', 'label subject', 
            'prediction rank', 'freshman']

DETAIL_RESULT_HEADER = ['ppsk', 'is_undergrad', 'college_identifier', 'major_identifier', 
                        'predicted_course_index', 'predicted_course_identifier',
                        'predicted_course_department', 'predicted_course_subject',
                        'label_number', 'label_course_index', 'label_course_identifier', 
                        'label_course_department', 'label_course_subject', 'label_rank_in_softmax']

TRAINED_MODELS_DIR = '../models'
DUMP_RESULT_DIR = '../results'

eval_batch_size = 4096

def load_my_model(model_name):
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_name) + '.json'
    weights_dir = os.path.join(TRAINED_MODELS_DIR, model_name)
    json_file = open(model_dir, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    lstm_model = model_from_json(loaded_model_json)
    lstm_model.load_weights(weights_dir)
    return lstm_model

def get_history_list(student_matrix):
    '''
    3D matrix 
    '''
    student_number = student_matrix.shape[0]
    semester_number = student_matrix.shape[1]
    course_dim = student_matrix.shape[2]
    zero_vec = np.zeros(course_dim, dtype='int8')
    history_total_list = []
    for i in range(student_number):
        history_list = []
        for j in range(semester_number):
            if (student_matrix[i, j, :] == zero_vec).all():
                break
            course_index = np.where(student_matrix[i, j, :] != 0)[0].tolist()
            history_list += student_matrix[i, j, course_index].tolist()
        history_total_list.append(history_list)
    return history_total_list

def first_zero_index(my_matrix, data_type='int16'):
    '''
    input: 2D matrix (timestep, vec)
    return: get the first zero vec index in axis 0
    '''
    timestep_num = my_matrix.shape[0]
    label_dim = my_matrix.shape[1]
    zero_vec = np.zeros(label_dim, dtype=data_type)
    for i in range(timestep_num):
        if (my_matrix[i, :] == zero_vec).all():
            return max(i - 1, 0)
    return timestep_num - 1

def count_not_zero(label):
    '''
    count how many non-zeros are there in a vector.
    '''
    counter = 0
    for i in range(label.shape[0]):
        if label[i] != 0:
            counter += 1
        else:
            break
    return counter

def isfreshman(matrix, data_type='int16'):
    '''
        2D matrix
        timestep, multihot
    '''
    assert len(matrix.shape) == 2
    timestep = matrix.shape[0]
    dimension = matrix.shape[1]
    zero_vec = np.zeros(dimension, dtype=data_type)
    if (matrix[0, :] == zero_vec).all():
        return True
    else:
        return False

def get_available_courses(eval_semester):
    '''
        return: {key: course_index(int), value: [identifier, department, subject]}
    '''
    return course_detail_dict[eval_semester]

def renormalize_probability(available_courses, softmax_vec):
    new_softmax_vec = np.copy(softmax_vec)
    for i in range(new_softmax_vec.shape[0]):
        if str((i + 1)) not in available_courses:
            new_softmax_vec[i] = 0.0
    total_value = np.sum(new_softmax_vec)
    new_softmax_vec = new_softmax_vec / total_value
    return new_softmax_vec

def run_on_evalset(eval_semester, model_name, x_course, x_major, x_grad, x_gpa, y_course, y_ppsk, 
                has_major=False, has_grad=False, has_gpa=False, has_bow=False, use_pca=False, detail_output=True):
    
    try:
        model = load_my_model(os.path.join(TRAINED_MODELS_DIR, model_name))
    except:
        print('There is no such model', model_name)
        print(os.path.join(TRAINED_MODELS_DIR, model_name))
        raise Exception()
    
    model.summary()
    available_courses = get_available_courses(eval_semester)

    over_all_correct = 0
    over_all_course_number = 0

    mrr_list = []
    detail_rows = []

    semester_results, undergrad_grad_recall = init_semester_dicts()
    college_recall = init_college_dicts()
    total_num_eval = x_course.shape[0]
    eval_run_counter = 0

    for x_small, x_major_multihot, x_grad_input, x_gpa_small, x_bow, y_small, ppsk_small in rnn_evaluate_batch_generator(x_course, x_major, x_grad, x_gpa, y_course, y_ppsk, use_pca, eval_batch_size):
        
        network_input = [x_small]
        # network_input = [x_bow]
        if has_major:
            network_input.append(x_major_multihot)
        if has_grad:
            network_input.append(x_grad_input)
        if has_gpa:
            network_input.append(x_gpa_small)
        if has_bow:
            network_input.append(x_bow)

        softmax_output = model.predict(network_input)[0]

        for i in range(x_small.shape[0]):
            # for each student
            current_ppsk = ppsk_small[i]
            if current_ppsk not in student_detail_dict:
                    raise Exception()
            graduate = student_detail_dict[current_ppsk][0]
            college = student_detail_dict[current_ppsk][1]
            if college == 'Other_EVCP_Programs' or college == 'UCB_Extension':
                continue
            index = first_zero_index(x_small[i, :, :]) # locate the target timestep 
            seq_length = index + 1 # length of 
            semester_in_input = seq_length - 1 # contains a BOS
            label_course_number = count_not_zero(y_small[i, :])
            current_label = y_small[i, 0:label_course_number].tolist() # current label courses

            softmax_vec = softmax_output[i, index, :] # probability over all courses
            softmax_vec = renormalize_probability(available_courses, softmax_vec)

            top_indexes = np.argsort(-softmax_vec)[0 : COURSE_NUM] # sorted index
            top_course_id = (top_indexes + 1).tolist()

            # prediction_number = label_course_number # precison
            prediction_number = 10 # recall@10, MRR@10
            course_prediction_list = top_course_id[0:prediction_number]

            # calculate overall recall & MRR
            current_hit = 0
            first_hit_rank = prediction_number + 1
            rank = 0
            for course in course_prediction_list:
                rank += 1
                if course in current_label:
                    over_all_correct += 1
                    current_hit += 1
                    first_hit_rank = min(rank, first_hit_rank)
            over_all_course_number += label_course_number
            if first_hit_rank == prediction_number + 1:
                MRR = 0.0
            else:
                MRR = 1 / float(first_hit_rank)
            mrr_list.append(MRR)

            if semester_in_input < 11:
                semester_results[semester_in_input][0] += current_hit
                semester_results[semester_in_input][1] += label_course_number
                semester_results[semester_in_input][2].append(MRR)
                undergrad_grad_recall[graduate][semester_in_input][0] += current_hit
                undergrad_grad_recall[graduate][semester_in_input][1] += label_course_number
                undergrad_grad_recall[graduate][semester_in_input][2].append(MRR)

            if detail_output: # break into colleges
                college_identifier = student_detail_dict[current_ppsk][1]
                college_recall[college_identifier][0] += current_hit
                college_recall[college_identifier][1] += label_course_number
                college_recall[college_identifier][2].append(MRR)

        eval_run_counter += eval_batch_size
        print('[{}/{}]Done'.format(min(eval_run_counter, total_num_eval), total_num_eval))
                    
    overall_recall = float(over_all_correct) / float(over_all_course_number)
    overall_mrr = np.mean(mrr_list)
    detailed_str = ''
    output_semester_dict = {}
    output_college_dict = {}
    for number, my_list in semester_results.items():
        if my_list[1] > 0:
            logging_str = 'Semester number {} Recall@10: {} MRR@10: {}'.format(number, my_list[0]/my_list[1], np.mean(my_list[2]))
            detailed_str += logging_str + '\n'
            output_semester_dict[number] = (my_list[0]/my_list[1], np.mean(my_list[2]))
    if detail_output:
        for college, my_list in college_recall.items():
            if my_list[1] > 0:
                logging_str = 'College: {} Recall@10 {} MRR@10 {}'.format(college, my_list[0]/my_list[1], np.mean(my_list[2]))
                detailed_str += logging_str + '\n'
                output_college_dict[college] = (my_list[0]/my_list[1], np.mean(my_list[2]))
    dump_detail_results(model_name, undergrad_grad_recall, output_college_dict)
    return overall_recall, overall_mrr, detailed_str

def init_semester_dicts():
    semester_number_recall = collections.OrderedDict()
    undergrad_grad_recall = {}
    undergrad_grad_recall['U'] = collections.OrderedDict()
    undergrad_grad_recall['G'] = collections.OrderedDict()
    for i in range(0, 11):
        semester_number_recall[i] = [0., 0., []]
        undergrad_grad_recall['U'][i] = [0., 0., []]
        undergrad_grad_recall['G'][i] = [0., 0., []]
    return semester_number_recall, undergrad_grad_recall

def init_college_dicts():
    college_recall = {}
    for key, value in college_dict.items():
        college_recall[key] = [0., 0., []]
    return college_recall

def dump_detail_results(model_name, semester_dict, college_dict):
    semester_name = model_name + '_semester.json'
    college_name = model_name + '_college.json'
    with open(os.path.join(DUMP_RESULT_DIR, semester_name), 'w', encoding='utf-8') as f:
        json.dump(semester_dict, f)
    with open(os.path.join(DUMP_RESULT_DIR, college_name), 'w', encoding='utf-8') as f:
        json.dump(college_dict, f)
    print('[Dumped] results into', DUMP_RESULT_DIR)