# -*- coding: utf-8 -*-
import numpy as np
import json
import csv
from random import *
from collections import OrderedDict
from bisect import bisect
from data_prepare_edw import dict_add_new_item

MAX_SEM = 19
MAX_COURSE = 13
MAX_COURSE_ALL = 74
COURSE_NUM = 9038 + 1
BOS_INDEX = COURSE_NUM
MAJOR_NUM = 253
MAX_MAJOR = 3
ENTRY_TYPE_NUM = 3
GRADE_TYPE = 8
#MAJOR_NUM = 259

random_generator = Random()
random_generator.seed(123)

ENROLLMENT_JSON_DIR = '../preprocess/enrollment_data.json' # course history
GRADE_JSON_DIR = '../preprocess/grade_data.json' # letter grade
GPA_JSON_DIR = '../preprocess/student_gpa.json' # float number gpa
COURSE_DICT_JSON_DIR = '../preprocess/edw_enrollment_course_dict.json' # index to course identifier
MAJOR_DICT_DIR = '../preprocess/major_dict.json' # index to major
MAJOR_DATA_DIR = '../preprocess/major_data.json' # major per semester
STUDENT_DETAIL_DIR = '../preprocess/student_detail.json' # gradstatus, college, division, department, major
COURSE_DETAIL_DIR = '../preprocess/course_detail_dict.json' # each semester, course and its department
ENTRY_DICT_DIR = '../preprocess/entry_dict.json' # each ppsk, corresponding entry type
GRADUATE_DICT_DIR = '../preprocess/graduate_status.json' # each ppsk ,corresponding grad status

#loaded from files
'''
    key: course index (str)
    value: course identifier (str)
'''
course_dict = OrderedDict()

'''
    key: ppsk
    value: key: semester_id (str)
           value: list of int
'''
enrollment_dict = OrderedDict() # ppsk and semester are strings, courses are numbers

'''
    key: ppsk
    value: key:semester_id (str)
           value: list of int
'''
grade_dict = OrderedDict()

'''
    key: major index (str)
    value: major identifier (str)
'''
major_dict = OrderedDict()

'''
    key: ppsk
    value: key: semester
           value: major
'''
major_data = OrderedDict()

'''
    key ppsk
    value: dict
        key: semester
        value: gpa
'''
gpa_dict = OrderedDict()

'''
    key ppsk
    value 1 for under 
          2 for grad
'''
graduate_status_dict = {}

'''
    key: ppsk (str)
    value: ['U/G', college, division, department, major]
'''
student_detail_dict = {}

'''
    key: department_identifier
    value: 1
'''
college_dict = {}

'''
    key: semester identifier (str)
    value: key: course index (str)
           value: [course identifier, department, subject]
'''
course_detail_dict = {}

'''
    key: ppsk (str)
    value: entry_type (int)
'''
entry_dict = {}

'''
    counting unique course numbers
'''
training_course_dict = {}
eval_course_dict = {}

with open(COURSE_DICT_JSON_DIR, 'r') as f:
    course_dict = json.load(f, object_pairs_hook=OrderedDict)

with open(ENROLLMENT_JSON_DIR, 'r') as f:
    enrollment_dict = json.load(f)

with open(GRADE_JSON_DIR, 'r') as f:
    grade_dict = json.load(f)

with open(MAJOR_DICT_DIR, 'r') as f:
    major_dict = json.load(f, object_pairs_hook=OrderedDict)

with open(MAJOR_DATA_DIR, 'r') as f:
    major_data = json.load(f)

with open(STUDENT_DETAIL_DIR, 'r') as f:
    student_detail_dict = json.load(f)

with open(COURSE_DETAIL_DIR, 'r') as f:
    course_detail_dict = json.load(f)

with open(ENTRY_DICT_DIR, 'r') as f:
    entry_dict = json.load(f)

with open(GPA_JSON_DIR, 'r') as f:
    gpa_dict = json.load(f)

with open(GRADUATE_DICT_DIR, 'r') as f:
    graduate_status_dict = json.load(f)

def sort_dict_by_semester(my_dict):
    temp_dict = OrderedDict()
    for ppsk, student_dict in my_dict.items():
        individual_dict = {}
        for semester_string, content in student_dict.items():
            semester_int = int(semester_string)
            if semester_int > 20082:
                individual_dict[semester_int] = content
        if len(individual_dict) > 0:
            temp_dict[ppsk] = OrderedDict(sorted(individual_dict.items()))
    return temp_dict

for ppsk, my_list in student_detail_dict.items():
    college_identifier = my_list[1]
    if college_identifier not in college_dict:
        college_dict[college_identifier] = 1

enrollment_dict = sort_dict_by_semester(enrollment_dict)
major_data = sort_dict_by_semester(major_data)
grade_dict = sort_dict_by_semester(grade_dict)
gpa_dict = sort_dict_by_semester(gpa_dict)

def get_ngram_datset(eval_semester):
    eval_semester = int(eval_semester)
    training_counter = 0
    evaluation_counter = 0
    evaluation_ppsk = []
    training_set = []
    eval_set = []
    eval_label = []
    for ppsk, student_dict in enrollment_dict.items():
        # ----------------------------Evaluation set--------------------------------
        if eval_semester in student_dict:
            x_eval_list = [(BOS_INDEX,)]
            # x_eval_list = []
            y_eval_courses = []
            for semester, courses in student_dict.items():
                if not (semester == eval_semester):
                    x_eval_list.append(tuple(courses))
                else:
                    y_eval_courses = tuple(courses)
                    break
            evaluation_ppsk.append(ppsk)
            eval_set.append(tuple(x_eval_list))
            eval_label.append(y_eval_courses)
            evaluation_counter += 1
        # ----------------------------Evaluation set--------------------------------

        # ----------------------------Training set--------------------------------
        sequence_list = [(BOS_INDEX,)]
        # sequence_list = []
        for semester, courses in student_dict.items():
            if semester < eval_semester:
                sequence_list.append(tuple(courses))
        training_set.append(tuple(sequence_list))
        training_counter += 1
        # ----------------------------Training set--------------------------------
    return training_set, eval_set, eval_label, evaluation_ppsk

def generate_course_vector(course_list, max_course):
    vector = np.zeros(max_course, dtype='int16')
    course_list_len = len(course_list)
    if course_list_len > max_course:
        sampled_rank = np.random.permutation(course_list_len)[0:max_course]
        course_list = [ course_list[i] for i in sampled_rank ]
        # course_list = random_generator.sample(course_list, course_list_len)
        # course_list = course_list[0:max_course]
    for i in range(len(course_list)):
        vector[i] = course_list[i]
    return vector

def generate_course_grade_vector(course_list, grade_list, max_course):
    course_vector = np.zeros(max_course, dtype='int16')
    grade_vector = np.zeros(max_course, dtype='int16')
    course_list_len = len(course_list)
    if course_list_len > max_course:
        sampled_rank = np.random.permutation(course_list_len)[0:max_course]
        course_list = [ course_list[i] for i in sampled_rank ]
        grade_list = [ grade_list[i] for i in sampled_rank ]
    for i in range(len(course_list)):
        course_vector[i] = course_list[i]
        grade_vector[i] = grade_list[i]
    return course_vector, grade_vector

def major_vec_helper(major_list):
    '''
    given a major list
    return a small numpy vec that length equals to MAX_MAJOR
    '''
    small_vec = np.zeros(MAX_MAJOR, dtype='int16')
    major_number = len(major_list)
    for i in range(major_number):
        small_vec[i] = major_list[i]
    return small_vec

def generate_major_vector(ppsk, sem_len, sem_num):
    '''
        sem_len: number of non-zero
    '''
    # shifted one semester
    major_vec = np.zeros((sem_num, MAX_MAJOR), dtype='int16')
    generated_sem_number = 0
    if ppsk in major_data:
        student_major_dict = major_data[ppsk]
        for semester, major_list in student_major_dict.items():
            if generated_sem_number == sem_len:
                break
            major_vec[generated_sem_number] = major_vec_helper(major_list)
            generated_sem_number += 1
    return major_vec

def generate_gpa_vector(ppsk, length, sem_num):
    gpa_vec = np.zeros(sem_num, dtype='float32')
    if ppsk in gpa_dict:
        counter = 0
        for semester, gpa in gpa_dict[ppsk].items():
            gpa_vec[counter] = gpa
            counter += 1
    for i in range(length, sem_num):
        gpa_vec[i] = 0.
    return gpa_vec

def get_rnn_dataset(eval_semester, max_semester=0, max_course_in =0, max_course_out=0, random_check=False):
    
    eval_semester = int(eval_semester)
    if not check_and_fill_major_data():
        print("Major data is not compatible with grade data!")
        raise Exception()

    # set up dimension settings for the dataset numpy array
    init_semester = MAX_SEM
    if max_semester > 0:
        init_semester = max_semester + 1 # add a <BOS> signal
    init_course_in_num = MAX_COURSE
    if max_course_in > 0:
        init_course_in_num = max_course_in
    init_course_out_num = MAX_COURSE
    if max_course_out > 0:
        init_course_out_num = max_course_out
    
    total_student_number = len(enrollment_dict)

    # assign memory for matrix that we will our dataset generate from
    '''
        dim == 3:
            (student_number, semester_number, course_number)
        dim == 2:
            (student_number, semester_number,)
    '''
    x_matrix = np.zeros((total_student_number, init_semester, init_course_in_num), dtype='int16')
    x_major = np.zeros((total_student_number, init_semester, MAX_MAJOR), dtype='int16')
    x_grad = np.zeros((total_student_number, init_semester), dtype='int16')
    x_gpa = np.zeros((total_student_number, init_semester), dtype='float32')

    y_matrix = np.zeros((total_student_number, init_semester, init_course_out_num), dtype='int16')
    
    x_eval_matrix = np.zeros((total_student_number, init_semester, init_course_in_num), dtype='int16')
    x_eval_major = np.zeros((total_student_number, init_semester, MAX_MAJOR), dtype='int16')
    x_eval_grad = np.zeros((total_student_number, init_semester), dtype='int16')
    x_eval_gpa = np.zeros((total_student_number, init_semester), dtype='float32')
    
    y_eval_matrix = np.zeros((total_student_number, MAX_COURSE), dtype='int16')

    sample_counter = 0 # records how many samples are in the dataset at last
    training_counter = 0 # records how many samples are in the training dataset
    evaluation_counter = 0 # records how many samples contains evaluation semester

    evaluation_ppsk = []

    for ppsk, student_dict in enrollment_dict.items():
        # slice the samples that has more timesteps than we want
        if len(student_dict) > init_semester - 1: # minus one due to the extra <BOS>
            continue

        # flag for if this sample contains eval_semester
        is_in_dataset = False

        # ----------------------------Evaluation set--------------------------------
        # only consider the semesters before eval_semester
        if eval_semester in student_dict: # to test course prediction result in this semester.
            x_eval_list = []
            y_eval_courses = []
            for semester, courses in student_dict.items():
                if not (semester == eval_semester):
                    x_eval_list.append(courses) # before the last semester
                else:
                    y_eval_courses = courses
                    for course in courses:
                        dict_add_new_item(eval_course_dict, course, 1) # to count availabel courses that semester
                    break
            eval_semester_number = len(x_eval_list)
            for i in range(eval_semester_number):
                x_eval_matrix[evaluation_counter][i + 1] = generate_course_vector(x_eval_list[i], init_course_in_num)
            x_eval_matrix[evaluation_counter][0] = generate_course_vector([BOS_INDEX], init_course_in_num)
            x_eval_major[evaluation_counter][1:] = generate_major_vector(ppsk, eval_semester_number, init_semester - 1)
            x_eval_grad[evaluation_counter, 0:eval_semester_number+1] = graduate_status_dict[ppsk] if ppsk in graduate_status_dict else 0
            x_eval_gpa[evaluation_counter][1:] = generate_gpa_vector(ppsk, eval_semester_number, init_semester - 1)
            y_eval_matrix[evaluation_counter] = generate_course_vector(y_eval_courses, MAX_COURSE)
            evaluation_ppsk.append(ppsk)
            evaluation_counter += 1
            is_in_dataset = True
        # ----------------------------Evaluation set--------------------------------

        # ----------------------------Training set--------------------------------
        # filter the courses before 2008 Fall and after evaluation semester
        history_list = []
        for semester, courses in student_dict.items():
            if int(semester) < int(eval_semester):
                history_list.append(courses)
        # ignore the students who has less than 2 semesters' records for training set
        if len(history_list) > 0:
            for course_list in history_list:
                for course in course_list:
                    dict_add_new_item(training_course_dict, course, 1) # training set unique course number
            x_list = [[BOS_INDEX]] + history_list[0:-1]
            y_list = list(history_list)
            semester_number = len(x_list) - 1
            x_major[training_counter][1:] = generate_major_vector(ppsk, semester_number, init_semester - 1)
            x_grad[training_counter, 0:semester_number + 1] = graduate_status_dict[ppsk] if ppsk in graduate_status_dict else 0
            x_gpa[training_counter, 1:] = generate_gpa_vector(ppsk, semester_number, init_semester - 1)
            for i in range(semester_number + 1):
                x_matrix[training_counter][i] = generate_course_vector(x_list[i], init_course_in_num)
                y_matrix[training_counter][i] = generate_course_vector(y_list[i], init_course_out_num)
            training_counter += 1
            is_in_dataset = True
        # ----------------------------Training set--------------------------------

        if is_in_dataset:
            sample_counter += 1

    # clipping
    x_matrix = x_matrix[0:training_counter, :, :]
    x_major = x_major[0:training_counter, :, :]
    x_grad = x_grad[0:training_counter, :]
    x_gpa = x_gpa[0:training_counter, :]
    y_matrix = y_matrix[0:training_counter, :, :]
    x_eval_matrix = x_eval_matrix[0:evaluation_counter, :, :]
    x_eval_major = x_eval_major[0:evaluation_counter, :, :]
    x_eval_grad = x_eval_grad[0:evaluation_counter, :]
    x_eval_gpa = x_eval_gpa[0:evaluation_counter, :]
    y_eval_matrix = y_eval_matrix[0:evaluation_counter, :]

    course_matrix = (x_matrix, x_eval_matrix)
    major_matrix = (x_major, x_eval_major)
    grad_matrix = (x_grad, x_eval_grad)
    gpa_matrix = (x_gpa, x_eval_gpa)
    target_matrix = (y_matrix, y_eval_matrix)

    if random_check:
        random_index = np.random.choice(training_counter, 1)
        print('Random checking: course matrix\n', x_matrix[random_index, :, :], 
            '\nmajor matrix\n', x_major[random_index, :, :], 
            '\ngraduate status\n', x_grad[random_index, :],
            '\ngpa\n',x_gpa[random_index, :],
            '\nlabel matrix\n', y_matrix[random_index, :, :])

    print("Dataset matrix generated. There are", evaluation_counter + training_counter, "samples in the data set, generated from", sample_counter, "students.")
    print('Training set size', training_counter, 'Evaluation set size', evaluation_counter)
    print('Training from', len(training_course_dict), 'unique courses, testing on', len(eval_course_dict), 'unique courses.')

    return course_matrix, major_matrix, grad_matrix, gpa_matrix, target_matrix, np.asarray(evaluation_ppsk)

def check_and_fill_entry_data():
    for ppsk, student_dict in enrollment_dict.items():
        if ppsk not in entry_dict:
            entry_dict[ppsk] = 0
    return True

def check_and_fill_major_data():
    sem_not_equal_num = 0
    not_equal_list = []
    global major_data
    for ppsk, student_dict in enrollment_dict.items():
        if ppsk not in major_data:
            raise Exception()
        else:
            # fill blanks
            pair_list = major_data[ppsk].items()
            semester_list = [ item[0] for item in pair_list ]
            major_list = [ item[1] for item in pair_list ]
            for semester, course_list in student_dict.items():
                if semester not in major_data[ppsk]:
                    index = bisect(semester_list, semester)
                    blank_filler = major_list[0]
                    if index > 0:
                        blank_filler = major_list[index - 1]
                    major_data[ppsk][semester] = blank_filler
            # sort it after done
            individual_dict = {}
            for semester_int, major_list in major_data[ppsk].items():
                if semester_int in student_dict:
                    individual_dict[semester_int] = major_list
            major_data[ppsk] = OrderedDict(sorted(individual_dict.items()))
            # make sure there is no empty
            student_major_dict = major_data[ppsk]
            major_sem_num = len(student_major_dict)
            course_sem_num = len(student_dict)
            if major_sem_num == course_sem_num:
                continue
            else:
                sem_not_equal_num += 1
                not_equal_list.append((student_dict, student_major_dict))
    if sem_not_equal_num == 0:
        return True
    return False

def get_median_number():
    course_numbers = []
    for ppsk, student_dict in enrollment_dict.items():
        for semester, course_list in student_dict.items():
            course_numbers.append(len(course_list))
    print('Median', np.median(course_numbers))

if __name__ == '__main__':
    course_matrix, major_matrix, grad_matrix, gpa_matrix, target_matrix, ppsk = get_rnn_dataset('20153', 12, MAX_COURSE, MAX_COURSE)
    print(np.mean(gpa_matrix[0], axis=0))