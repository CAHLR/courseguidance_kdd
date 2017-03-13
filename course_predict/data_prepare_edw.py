import numpy as np
import os
import csv
import json
from collections import OrderedDict
from random import shuffle

OUTPUT_FLAG = True

# directory constants
EDW_SRC_DIR = "../data"
EDW_OUT_DIR = "../preprocess"

EDW_GRADE_FILENAMES = ['student_grade_data_2008.tsv',
                'student_grade_data_2009.tsv',
                'student_grade_data_2010.tsv',
                'student_grade_data_2011.tsv',
                'student_grade_data_2012.tsv',
                'student_grade_data_2013.tsv',
                'student_grade_data_2014.tsv',
                'student_grade_data_2015.tsv']

# filter settings
PRIME_COURSE_ONLY = True
UNDERGRADUATE_INCLUDE = True
GRADUATE_INCLUDE = True

# dictionaries
'''
    key: index(str)
    value: depart, course number concat(using '_')
'''
course_dict = {}

'''
 key:ppsk value: [is_undergrad, college_identifier, major_identifier]
'''
student_detail_dict = {}

'''
    key: semester(int)
    value:dict
    key: course_index(int)
    value: [course_identifier, department_name, subject_name]
'''
course_detail_dict = {}

'''
    key: ppsk
    value: dict
    key: semester
    value: course list
'''
student_dict = OrderedDict()

'''
    key: ppsk
    value: dict
    key: semester
    value: grade list
'''
grade_dict = {}
subtype_dict = {} #Grade Subtype Desc

'''
    key: ppsk
    value: dict
        key: semester
        value: [(score, credit),]
'''
gpa_dict = {}

'''
    key: ppsk(str)
    value: 1(int)
'''
undergrad_student_dict = {}

'''
    key: ppsk(str)
    value: 1(int)
'''
grad_student_dict = {}

'''
    key: course_identifier(str)
    value: 1
'''
grade_course_dict = {}

'''
    key: course index
    value: 1(int)
    all courses 
'''
demand_course_dict = {}

'''
    key: course index
    value: 1(int)
    prime courses 
'''
prime_course_dict = {}

'''
    key: major index
    value: major name
'''
major_dict = {}

'''
    key: college name
    value: student_number
'''
college_dict = {}


'''
    key: department name
    value: 1
'''
department_dict = {}

'''
    key: ppsk
    value: dict
    key: semester
    value: list of major index
'''
student_major_dict = OrderedDict()

'''
    key: ppsk (str)
    value: entry_type (int)
'''
entry_dict = {}
entry_type_lookup = {}

'''
    key: ppsk(str)
    value: 1 for undergrad
           2 for grad
'''
graduate_status_dict = {}

# basic statistics
total_student_number = 0
max_number_course_taken_in_one_semester = 0
max_number_semester_for_one_student = 0


def dict_add_new_item(my_dict, key, value=None):
    if key not in my_dict:
        if value == None:
            my_dict[key] = len(my_dict) + 1
        else:
            my_dict[key] = value

def get_csv_row_content(header, row, col_name):
    index = header.index(col_name)
    return row[index]

def is_prime_course(header_of_row, row):
    if PRIME_COURSE_ONLY:
        prime_str = get_csv_row_content(header_of_row, row, 'Offering Type Desc')
        return prime_str == 'Primary'
    else:
        return True

def is_filtered_student(header_of_row, row):
    if UNDERGRADUATE_INCLUDE and GRADUATE_INCLUDE:
        return True
    elif UNDERGRADUATE_INCLUDE:
        return get_csv_row_content(header_of_row, row, 'undergraduate / graduate status') == "Undergraduate"
    elif GRADUATE_INCLUDE:
        return get_csv_row_content(header_of_row, row, 'undergraduate / graduate status') == 'Graduate'

def process_course_demand_row(header, row):
    year_semester = get_csv_row_content(header, row, 'Semester Year Name Concat')
    semester_id = int(get_semester_identifier(year_semester))
    course_number = get_csv_row_content(header, row, 'Course Number')
    subject_name = get_csv_row_content(header, row, 'Course Subject Short Nm')
    course_identifier = ' '.join([subject_name, course_number]).replace(' ', '_')
    demand_course_dict[course_identifier] = 1
    if is_prime_course(header, row):
        dict_add_new_item(course_dict, course_identifier)
        course_index = course_dict[course_identifier]
        dict_add_new_item(prime_course_dict, course_identifier, 1)
        # percentatge

def generate_course_dict(in_filename, out_index_course_filename):
    input_filename = os.path.join(EDW_SRC_DIR, in_filename)
    output_filename = os.path.join(EDW_OUT_DIR, out_index_course_filename)
    csv_header = []
    with open(input_filename, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        is_header = True
        for row in reader:
            if is_header:
                is_header = False
                csv_header = row
            else:
                process_course_demand_row(csv_header, row)
        if OUTPUT_FLAG:
            with open(output_filename, 'w', encoding="utf-8") as f:
                inv_dict = { v: k for k, v in course_dict.items() }
                json.dump(OrderedDict(sorted(inv_dict.items())), f)
    return len(course_dict)

def get_semester_identifier(my_string):
    '''
        input: year sememster concat e.g. 2014 FALL
        output: str(year semester(int) concat)
    '''
    my_list = my_string.strip().split()
    year = my_list[0]
    sem = my_list[1]
    sem_integer = "0"
    if sem == "Spring":
        sem_integer = "1"
    elif sem == "Summer":
        sem_integer = "2"
    else:
        sem_integer = "3"
    res_string = "".join([year, sem_integer])
    return int(res_string)

def get_course_id(subject, num):
    course_identifier = ' '.join([subject, num]).replace(' ', '_')
    if course_identifier in course_dict:
        return course_dict[course_identifier], course_identifier
    else:
        # print(course_identifier)
        return -1

def process_student_grade_row(header, row):
    if is_filtered_student(header, row):
        sem_id = get_semester_identifier(get_csv_row_content(header, row, 'Semester Year Name Concat')) # integer semester
        ppsk = get_csv_row_content(header, row, ' Student Identifier(ppsk)')
        
        if ppsk == "-1" or int(sem_id) < 20083 or not (ppsk in student_major_dict):
            return
        
        if get_csv_row_content(header, row, 'undergraduate / graduate status') == "Undergraduate":
            undergrad_student_dict[ppsk] = 1
        elif get_csv_row_content(header, row, 'undergraduate / graduate status') == "Graduate":
            grad_student_dict[ppsk] = 1
        
        course_id, course_identifier = get_course_id(get_csv_row_content(header, row, 'Course Subject Short Nm'), get_csv_row_content(header, row, 'Course Number'))
        
        if course_id == -1:
            return
        
        department_name = get_csv_row_content(header, row, 'Crs Academic Dept Short Nm')
        grade_sub_type_desc = get_csv_row_content(header, row, 'Grade Subtype Desc')
        grade_points = get_csv_row_content(header, row, 'Grade Points Nbr')
        credit_hours = get_csv_row_content(header, row, 'Student Credit Hrs Nbr')
        dict_add_new_item(subtype_dict, grade_sub_type_desc)
        grade_sub_type_index = subtype_dict[grade_sub_type_desc]

        global student_dict
        dict_add_new_item(student_dict, ppsk, OrderedDict())
        dict_add_new_item(student_dict[ppsk], sem_id, [])
        skip_flag = False     
        if course_id not in student_dict[ppsk][sem_id]:
            student_dict[ppsk][sem_id].append(course_id)
        else:
            skip_flag = True

        # course grade
        global grade_dict       
        dict_add_new_item(grade_dict, ppsk, OrderedDict())
        dict_add_new_item(grade_dict[ppsk], sem_id, [])
        if not skip_flag:
            grade_dict[ppsk][sem_id].append(grade_sub_type_index)

        # course gpa
        global gpa_dict
        dict_add_new_item(gpa_dict, ppsk, OrderedDict())
        dict_add_new_item(gpa_dict[ppsk], sem_id, [])
        if not skip_flag:
            gpa_dict[ppsk][sem_id].append((grade_points, credit_hours))

        # course detail
        dict_add_new_item(course_detail_dict, sem_id, {})
        dict_add_new_item(course_detail_dict[sem_id], int(course_id), [course_identifier, department_name, get_csv_row_content(header, row, 'Course Subject Short Nm')])

def calculate_gpa():
    result = {}
    for ppsk, individual_dict in gpa_dict.items():
        dict_add_new_item(result, ppsk, OrderedDict())
        for semester, pair_list in individual_dict.items():
            total_grade = 0.
            total_credit = 0.
            for pair in pair_list:
                try:
                    current_grade = float(pair[0])
                    current_credit = float(pair[1])
                    total_grade += current_grade * current_credit
                    total_credit += current_credit
                except:
                    total_grade += 4.0
                    total_credit += 1.0
            if total_grade > 0.1:
                gpa = round(float(total_grade / total_credit), 1)
                dict_add_new_item(result[ppsk], semester, gpa)
    return result

def generate_student_grade_json(filename, out_grade_filename, out_detail_filename, out_gpa_filename):
    for course_file in EDW_GRADE_FILENAMES:
        with open(os.path.join(EDW_SRC_DIR, course_file), newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")
            is_header = True
            csv_header = []
            for row in reader:
                if is_header:
                    is_header = False
                    csv_header = row
                else:
                    if is_prime_course(csv_header, row):
                        process_student_grade_row(csv_header, row)
    # after all files done
    student_filtered_num = 0
    max_sem = 0
    max_course = 0
    global student_dict
    global grade_dict
    for ppsk, individual_dict in student_dict.items():
        student_filtered_num += 1
        max_sem = max(max_sem, len(individual_dict))
        for semester_id, course_ids in individual_dict.items():
            # get course id & sub type
            grade_list = grade_dict[ppsk][semester_id]
            course_list_len = len(course_ids)
            max_course = max(max_course, course_list_len)
            assert course_list_len == len(grade_list)
            # shuffle two lists
            combined = list(zip(course_ids, grade_list))
            shuffle(combined)
            course_ids_shuffled, grade_list_shuffled = zip(*combined)
            student_dict[ppsk][semester_id] = course_ids_shuffled
            grade_dict[ppsk][semester_id] = grade_list_shuffled

    my_gpa = calculate_gpa()
            
    if OUTPUT_FLAG:
        output_detail_filename = os.path.join(EDW_OUT_DIR, out_detail_filename)
        output_grade_filename = os.path.join(EDW_OUT_DIR, out_grade_filename)
        output_grade_sub_filename = os.path.join(EDW_OUT_DIR, 'grade_sub_dict.json')
        output_gpa_filename = os.path.join(EDW_OUT_DIR, out_gpa_filename)
        with open(os.path.join(EDW_OUT_DIR, filename), 'w', encoding="utf-8") as f:
            json.dump(student_dict, f)
        with open(output_detail_filename, 'w', encoding='utf-8') as f:
            json.dump(course_detail_dict, f)
        with open(output_grade_filename, 'w', encoding='utf-8') as f:
            json.dump(grade_dict, f)
        with open(output_grade_sub_filename, 'w', encoding='utf-8') as f:
            json.dump(subtype_dict, f)
        with open(output_gpa_filename, 'w', encoding='utf-8') as f:
            json.dump(my_gpa, f)

    return student_filtered_num, max_sem, max_course

def get_major_index(major_id):
    if major_id in major_dict:
        return major_dict[major_id]
    else:
        int_id = len(major_dict) + 1
        major_dict[major_id] = int_id
        return int_id

major_data_header = ['ppsk', 'undergrad.status', 'term.majors', 'year.majors',
                    'Count of Students',
                    'college',
                    'division',
                    'department',
                    'major',
                    'exam.units']

def process_entry_row(header, row):
    ppsk = get_csv_row_content(header, row, 'ppsk')
    entry_type = get_csv_row_content(header, row, 'entry.type')
    dict_add_new_item(entry_type_lookup, entry_type)
    dict_add_new_item(entry_dict, ppsk, entry_type_lookup[entry_type])

def process_major_row(header, row):
    ppsk = get_csv_row_content(header, row, 'ppsk')
    is_undergrad = get_csv_row_content(header, row, 'undergrad.status')
    if is_undergrad == 'U':
        graduate_status_index = 1
    else:
        graduate_status_index = 2
    term = get_csv_row_content(header, row, 'term.majors')
    year = get_csv_row_content(header, row, 'year.majors')
    college = get_csv_row_content(header, row, 'college')
    division = get_csv_row_content(header, row, 'division')
    department = get_csv_row_content(header, row, 'department')
    major = get_csv_row_content(header, row, 'major')
    
    # deal with major dict
    college_identifier = college.replace(' ', '_')
    division_identifier = division.replace(' ', '_')
    department_identifier = department.replace(' ', '_')
    major_identifier = major.replace(' ', '_')
    global college_dict
    dict_add_new_item(college_dict, college_identifier, 1)
    college_dict[college_identifier] += 1
    dict_add_new_item(department_dict, department_identifier, 1)
    # deal with student major info
    sem_string = ' '.join([year, term])
    sem_id = get_semester_identifier(sem_string)
    sem_id = int(sem_id)
    if ppsk == "-1" or sem_id < 20083 or major == 'Summer_Session_Undeclared':
        return

    major_int = get_major_index(major_identifier)
    dict_add_new_item(graduate_status_dict, ppsk, graduate_status_index)

    student_detail_dict[ppsk] = [is_undergrad, college_identifier, division_identifier, department_identifier, major_identifier]
    
    if ppsk not in student_major_dict:
        student_major_dict[ppsk] = OrderedDict()

    if sem_id not in student_major_dict[ppsk]:
        student_major_dict[ppsk][sem_id] = []
    
    student_major_dict[ppsk][sem_id].append(major_int)

def generate_major_dict(in_filename, dict_filename, out_filename):
    input_filename = os.path.join(EDW_SRC_DIR, in_filename)
    dict_major_filename = os.path.join(EDW_OUT_DIR, dict_filename)
    output_filename = os.path.join(EDW_OUT_DIR, out_filename)
    with open(input_filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        is_header = True
        csv_header = []
        for row in reader:
            if is_header:
                is_header = False
                csv_header = row
            else:
                process_major_row(csv_header, row)
    if OUTPUT_FLAG:
        with open(os.path.join(EDW_OUT_DIR, 'student_detail.json'), 'w', encoding='utf-8') as f:
            json.dump(student_detail_dict, f)
        with open(os.path.join(EDW_OUT_DIR, 'graduate_status.json'), 'w', encoding='utf-8') as f:
            json.dump(graduate_status_dict, f)
        with open(dict_major_filename, 'w', encoding='utf-8') as f:
            inv_dict = { v:k for k, v in major_dict.items() }
            json.dump(OrderedDict(sorted(inv_dict.items())), f)
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(student_major_dict, f)
    return len(major_dict), len(college_dict), len(department_dict)

def generate_entry_dict(in_filename, out_filename):
    input_filename = os.path.join(EDW_SRC_DIR, in_filename)
    output_filename = os.path.join(EDW_OUT_DIR, out_filename)
    with open(input_filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        is_header = True
        csv_header = []
        for row in reader:
            if is_header:
                csv_header = row
                is_header = False
            else:
                process_entry_row(csv_header, row)
    if OUTPUT_FLAG:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(entry_dict, f)

def get_percentage_result(my_dict):
    result = {}
    total_number = 0
    for key, value in my_dict.items():
        total_number += value
        result[key] = value
    for key, value in result.items():
        result[key] = value / total_number * 100
    return result

def main():
    unique_course_number = generate_course_dict('course_demand_data.tsv', 'edw_enrollment_course_dict.json')
    print('There are', unique_course_number, 'courses.')
    major_num, college_num, department_num = generate_major_dict('student_majors_data.tsv', 'major_dict.json', 'major_data.json')
    print('major num', major_num, 'college num', college_num, 'department num', department_num)
    total_student_number, max_number_semester_for_one_student, max_number_course_taken_in_one_semester = generate_student_grade_json('enrollment_data.json', 'grade_data.json', 'course_detail_dict.json', 'student_gpa.json')
    print('total student number', total_student_number, 'max semester', max_number_semester_for_one_student, 'max course', max_number_course_taken_in_one_semester)

if __name__ == '__main__':
    main()