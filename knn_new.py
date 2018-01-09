
from numpy import *
import operator
from os import listdir

TRAINING_PATH = 'digits/trainingDigits/'
TEST_PATH = 'digits/testDigits/'

TRAINING_RESULT = []
TEST_RESULT = []

def matrix_by_file(file):
	martix = zeros((1, 1024))
	with open(file) as lines:
		for line_num, line in enumerate(lines):
			line_count = len(line) - 1
			for i in range(line_count):
				martix[0, line_num * line_count + i] = int(line[i])
	return martix

def get_training_matrix():
	return get_martrix(TRAINING_PATH, TRAINING_RESULT)

def get_test_matrix():
	return get_martrix(TEST_PATH, TEST_RESULT)

def get_martrix(path, digit_list):
	training_file_list = listdir(path)
   	
	files_count = len(training_file_list)
	training_matrix = zeros((files_count, 1024))
	test_martix = zeros((files_count, 1024))

	for index, filename in enumerate(training_file_list):
		digit = filename.split('.')[0].split('_')[0]
		test_martix[index, :] = matrix_by_file(path + filename)
		digit_list.append(digit)
	return test_martix

def knn(test, train, k):

	training_matrix_lines = train.shape[0]
	test = tile(test, (training_matrix_lines, 1)) 

	distances = (((test - train) ** 2).sum(axis=1)) ** 0.5 
	sorted_distances_index = distances.argsort()

	num_count = {}
	max_count = -1
	max_digit = -1

	for i in range(k):
		index = sorted_distances_index[i]
		digit = TRAINING_RESULT[index]
		num_count[digit] = num_count.get(digit, 0) + 1
		
	for key, value in num_count.iteritems():
   	 	if max_count < num_count[key]:
   	 		max_digit = key
   	 		max_count = num_count[key]

   	return max_digit

def run():
	training_matrix = get_training_matrix()
	test_matrix_list = get_test_matrix()
	
	wrong = 0
	total = len(test_matrix_list)

	for idx, test_matrix in enumerate(test_matrix_list):
		test_digit = TEST_RESULT[idx]
		train_digit = knn(test_matrix, training_matrix, 3)
		if test_digit != train_digit :
			wrong = wrong + 1

	print('guss rate: ', (total - wrong) / float(total))

if __name__ == '__main__':
    run()

