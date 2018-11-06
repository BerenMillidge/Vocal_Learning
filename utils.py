# Utils code file for Vocal learning papers
# Author: Beren Millidge
# Date: Summer 2018



from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle



def euclidean_distance(center, point):
	if len(center)!=len(point):
		raise ValueError('Point and center must have same dimensionality')
	total = 0
	for i in xrange(len(center)):
		total += (center[i] - point[i])**2
	return np.sqrt(total)


def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def tuple_add(tup1, tup2):
	if len(tup1) != len(tup2):
		print len(tup1), len(tup2)
		raise ValueError('Length of the tuples to be added must be the same')
	l = []
	for i in xrange(len(tup1)):
		l.append(tup1[i] + tup2[i])
	return tuple(l)

def tuple_divide_scalar(tup1, s):
	l = []
	for i in xrange(len(tup1)):
		l.append(tup1[i]/s)
	return tuple(l)

def tuple_multiply(tup1, tup2):
	if len(tup1) != len(tup2):
		print len(tup1), len(tup2)
		raise ValueError('Length of the tuples to be added must be the same')
	l = []
	for i in xrange(len(tup1)):
		l.append(tup1[i] * tup2[i])
	return tuple(l)



def select_random_point(mat):
	h,w,ch = mat.shape
	selected = False
	while selected != True:
		height = int(h * np.random.uniform(low=0, high=1))
		width = int(w*np.random.uniform(low=0, high=1))
		if check_proposed_points((height, width), h,w):
			selected=True
	return height,width

def swap_points(mat):
	rand1 = select_random_point(mat)
	rand2 = select_random_point(mat)
	temp = mat[rand1]
	mat[rand1] = mat[rand2]
	mat[rand2] = temp
	return mat

def randomise_swap_points(mat, N):
	new_mat = np.copy(mat)
	for i in xrange(N):
		new_mat = swap_points(new_mat)
	return new_mat



def select_target(mat):
	height, width = select_random_point(mat)
	return mat[height][width]



def select_random_edge_point(mat):
	h,w,ch = mat.shape
	edge = 4*np.random.uniform(low=0, high=1)
	if edge<=1:
		rand = int(h*np.random.uniform(low=0, high=1))
		return (rand,0)
	if edge>1 and edge<=2:
		rand = int(h*np.random.uniform(low=0, high=1))
		return (rand, (w-1))
	if edge>2 and edge<=3:
		rand = int(w*np.random.uniform(low=0, high=1))
		return (0, rand)
	if edge>3 and edge<=4:
		rand = int(w*np.random.uniform(low=0, high=1))
		return ((h-1), rand)


def check_proposed_points(points, height,width):
	h,w = points
	if h>=0 and h<height:
		if w>= 0 and w<width:
			return True
	return False


def absolute_diff(p1,p2):
	if len(p1)!=len(p2):
		raise ValueError('Points to be compared must be of same dimension')
	total = 0
	for i in xrange(len(p1)):
		total += np.abs(p1[i] - p2[i])
	return total/len(p1)



def power_law_sample(alpha):
	samps = np.random.uniform(low=0, high=1, size=1)
	return np.power((1-samps), (-1/alpha-1))

def t_test(randoms, gradients):
	t,prob = scipy.stats.ttest_ind(randoms, gradients, equal_var=False)
	return t,prob

def t_test_results(rands,gradients,levys):
	print "t-test rands gradients"
	t,prob = t_test(rands, gradients)
	print t 
	print prob
	print "t-test rands levys"
	t,prob = t_test(rands, levys)
	print t
	print prob
	print "t-test gradients levys"
	t,prob = t_test(gradients, levys)
	print t
	print prob


def clip_vector(vect, max_val=255, min_val=0):
	for i in xrange(len(vect)):
		if vect[i] < min_val:
			vect[i] = min_val
		if vect[i] > max_val:
			vect[i] = max_val
	return vect

def tuple_add(tup1, tup2):
	l = []
	for i in xrange(len(tup1)):
		l.append(tup1[i] + tup2[i])
	return tuple(l)

def choose_constant_coords(mat, prob):
	h,w, ch = mat.shape
	coords = []
	for i in xrange(h-1):
		for j in xrange(w-1):
			rand = np.random.uniform(low=0, high=1)
			if rand < prob:
				coords.append((i,j))
	return coords

