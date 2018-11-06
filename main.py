# Main code file for Vocal learning papers
# Author: Beren Millidge
# Date: Summer 2018


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy
import scipy.stats
import os
from utils import *
from plots import *
from masks import *
from strategies import *

def create_random_N_dimensional_matrix(height, width,N):
	mat = np.zeros((height, width,N))
	for i in xrange(height):
		for j in xrange(width):
				for k in xrange(N):
					mat[i][j][k] = np.random.uniform(low=0, high=1) *255.
	return mat



def create_random_colour_matrix(height, width):
	mat = np.zeros((height, width,3))
	for i in xrange(height):
		for j in xrange(width):
			mat[i][j][0] = (np.random.uniform(low=0, high=1) * 255.)
			mat[i][j][1] = np.random.uniform(low=0, high=1) * 255.
			mat[i][j][2] = np.random.uniform(low=0, high=1) * 255.
	return mat


def average_point(mat,center,px_radius, image_height, image_width, dim=3):
	x,y = center
	number = 0
	total = np.zeros((dim))
	for i in xrange(1+(px_radius*2)):
		for j in xrange(1+(px_radius*2)):
			xpoint = x - px_radius + i
			ypoint = y - px_radius + j
			if xpoint >=0 and xpoint < image_height:
				if ypoint >=0 and ypoint <image_width:
					if euclidean_distance(center, (xpoint, ypoint)) <=px_radius:
						total += mat[xpoint][ypoint]
						number+=1

	return tuple(total/number)

def random_average_point(mat, center, px_radius, image_height, image_width, copy_prob):
	x,y = center
	green_total = 0
	red_total= 0
	blue_total = 0
	number = 0
	for i in xrange(1+(px_radius*2)):
		for j in xrange(1+(px_radius*2)):
			xpoint = x - px_radius + i
			ypoint = y - px_radius + j
			if xpoint >=0 and xpoint < image_height:
				if ypoint >=0 and ypoint <image_width:
					if euclidean_distance(center, (xpoint, ypoint)) <=px_radius:
						rand = np.random.uniform(low=0, high=1)
						if rand <=copy_prob and mat[xpoint][ypoint][0] !=0: # check if not zero!
							green_total+= mat[xpoint][ypoint][0]
							red_total+= mat[xpoint][ypoint][1]
							blue_total+=mat[xpoint][ypoint][2]
							number+=1

	if number == 0:
		return mat[x][y]

	return (green_total/number, red_total/number, blue_total/number)



def update_point(mat, center, px_radius, image_height, image_width, learning_rate=0.0001):
	x,y = center
	currents = mat[x][y]
	new_points = average_point(mat, center, px_radius, image_height, image_width)
	diff = new_points - currents
	currents = currents + (learning_rate*diff)
	for i in xrange(len(currents)):
		if currents[i] < 0:
			currents[i] = 0
	return currents


def matrix_average_step(mat, average_radius, copy=True, random_multiplier=None, dim=3, noise_sigma=None, constant_coords=None, constant_value = None):
	if len(mat.shape)!=3 or mat.shape[2]!=3:
		raise ValueError('Matrix must be 2d colour image with 3 channels in format h,w,ch')

	height,width, channels = mat.shape
	if not copy:
		new_mat = mat
	if copy:
		new_mat = np.copy(mat)
	for i in xrange(height):
		for j in xrange(width):
			if noise_sigma is None:
				new_mat[i][j] = average_point(mat, (i,j), average_radius, height,width,dim=dim)
			if noise_sigma is not None:
				noise = np.random.normal(loc=0, scale=noise_sigma, size=3)
				print noise
				average = average_point(mat, (i,j), average_radius, height, width, dim=dim)
				print average
				addition =  tuple_add(average, noise)
				print addition
				new_mat[i][j] = addition

	if random_multiplier is not None:
		rand = create_random_mask((height,width,channels), random_multiplier)
		new_mat = new_mat + rand
	
	if constant_coords is not None and constant_value is not None:
		if isinstance(constant_coords, list) and isinstance(constant_value, list):
			print "attempting to do the coords as list"
			assert len(constant_coords) == len(constant_value), 'Lengths of constant coords and values must be the same.'
			for i in xrange(len(constant_value)):
				new_mat[constant_coords[i]] = constant_value[i]
		else:
			new_mat[constant_coords] = constant_value
			print new_mat[constant_coords]
	return new_mat

def matrix_update_step(mat, radius, copy=True, learning_rate=0.1):

	if len(mat.shape)!=3 or mat.shape[2]!=3:
		raise ValueError('Matrix must be 2d colour image with 3 channels in format h,w,ch')

	height,width, channels = mat.shape
	if not copy:
		new_mat = mat
	if copy:
		new_mat = np.copy(mat)
	for i in xrange(height):
		for j in xrange(width):
			new_mat[i][j] = update_point(mat, (i,j), radius, height,width,learning_rate=learning_rate)

	return new_mat

def simple_test_test():
	orig_mat = create_random_colour_matrix(3,3)
	for i in xrange(1):
		print "run ", i
		orig_mat = matrix_average_step(orig_mat, 2)


def get_gradient_matrix(N=20, radius=5, plot=True, save_name=None,save_after=None,dim=3):
	orig_mat = create_random_colour_matrix(50,50)
	for i in xrange(N):
		orig_mat = matrix_average_step(orig_mat, radius)
		if save_after is not None:
			if i % save_after ==0:
				np.save(save_name+ '_' + str(i), orig_mat)

	if plot:
		plt.imshow(orig_mat)
		plt.show()

	if save_name:
		np.save(save_name, orig_mat)

	return orig_mat


def random_walk_step(mat, initial_point, step_size):
	sh,sw = initial_point
	h,w,ch = mat.shape
	valid=False
	#init coords to always be wrong
	coords = (-5,-5)
	while valid is False:
		direction = int(8*np.random.uniform())
		if direction == 0:
			coords =  sh+step_size, sw-step_size
		if direction==1:
			coords =  sh+step_size, sw
		if direction==2:
			coords =  sh+step_size, sw+step_size
		if direction==4:
			coords =  sh, sw+step_size
		if direction==5:
			coords =  sh-step_size, sw+step_size
		if direction==6:
			coords =  sh-step_size, sw
		if direction==7:
			coords =  sh-step_size, sw-step_size
		if direction==8:
			coords =  sh, sw-step_size

		if check_proposed_points(coords, h,w) is True:
			valid=True

	return coords

def immediate_gradient_step(ideal, center, mat):
	
	best_diff = 99999
	ch,cw = center
	best_coords = None
	h,w,channels = mat.shape


	for i in xrange(2): 
		for j in xrange(2):
			xpoint = ch+i -1
			ypoint = cw + j -1
			if xpoint >=0 and xpoint<=w:
				if ypoint>=0 and ypoint<=h:
					val = mat[xpoint][ypoint]
					diff = euclidean_distance(ideal, val)
					if diff<best_diff:
						best_diff=diff
						best_coords = (xpoint, ypoint)
	if best_coords == center:
		coords = random_walk_step(mat, center,1)
		diff = euclidean_distance(ideal, mat[coords])
		return coords, diff

	return best_coords, best_diff

def immediate_aversion_step(ideal, center, mat):
	start_diff = 0
	ch, cw = center
	aversive_coords = None
	h,w,channels = mat.shape
	print "Center: " + str(center)

	for i in xrange(3):
		for j in xrange(3):
			xpoint = ch + i -1
			ypoint = cw + j -1
			if xpoint >=0 and xpoint<w:
				if ypoint>=0 and ypoint<h:
					print xpoint, ypoint
					val = mat[xpoint][ypoint]
					diff = euclidean_distance(ideal, val)
					if diff > start_diff:
						start_diff = diff
						aversive_coords = (xpoint, ypoint)

	if aversive_coords == center or aversive_coords == (0,0):
		coords = random_walk_step(mat, center,2)
		diff = euclidean_distance(ideal, mat[coords])
		return coords, diff


	new_coords = aversive_coords
	if check_proposed_points(new_coords, h,w) is False:
		random_walk_step(mat, center, 1)
	diff = euclidean_distance(ideal, mat[new_coords])
	print "new coords: " + str(new_coords)
	return new_coords, diff

def position_base_on_path(path, base):
	new_base = np.copy(base)
	h,w = path.shape
	for i in xrange(h):
		for j in xrange(w):
			if path[i][j] >1:
				new_base[i][j] = 255.
	return new_base

def gradient_search_till_atop(mat, less_diff=0.01, save_name=None, plot=False,return_base=False, gradient_anim=True):

	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Matrix must be a colour image 3dimensional with 3rd dimension 3 colour channels')
	ideal_coords = select_random_point(mat)
	ideal = mat[ideal_coords]
	position = select_random_edge_point(mat)
	diffs = []
	coords = []

	h,w,ch = mat.shape
	diff = 100000

	tries=0
	max_tries = 1000

	while position != ideal_coords and tries <= max_tries:
		new_coords, diff = immediate_gradient_step(ideal, position,mat)
		diffs.append(diff)
		coords.append(new_coords)
		position = new_coords
		tries +=1

	if save_name is not None:
		save((diffs, coords), save_name)

	if gradient_anim and save_name is not None:
		slides = plot_anim_path(coords, h, w, ideal_coords, position, base=mat)
		sname = save_name + '_animation_slides'
		np.save(sname, slides)

	base = None
	if plot:
		base = plot_path(coords, h,w, base=mat)

	if return_base is True:
		return diffs, coords ,base
	return diffs, coords

def random_walk_till_atop(mat, less_diff=0.1, step_size=1,save_name=None, plot=False, plot_animation=False,return_base=False, gradient_anim=False):
	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Matrix must be a colour image 3dimensional with 3rd dimension 3 colour channels')
	ideal_coords = select_random_point(mat)
	ideal = mat[ideal_coords]
	position = select_random_edge_point(mat)
	diffs = []
	coords = []

	h,w,ch = mat.shape
	diff = 100000

	tries=0
	max_tries = 1000

	while position != ideal_coords and tries <= max_tries:

		new_coords = random_walk_step(mat,position, step_size=step_size)
		diff = euclidean_distance(mat[new_coords], ideal)
		diffs.append(diff)
		coords.append(new_coords)
		position = new_coords
		tries +=1

	if save_name is not None:
		save((diffs, coords), save_name)

	if gradient_anim and save_name is not None:
		slides = plot_anim_path(coords, h, w, ideal_coords, position, base=mat)
		sname = save_name + 'animation_slides'
		np.save(sname, slides)

	base = None
	if plot:
		base = plot_path(coords, h,w, base=mat)
	if plot_animation is True:
		anim_slides = plot_anim_path(mat, coords, h,w,ideal, position)
	print len(coords)
	print len(diffs)
	if return_base is True:
		return diffs, coords ,base

	return diffs, coords


def aversive_gradient_search_till_atop(mat, less_diff=0.01, save_name=None, plot=False,return_base=False, gradient_anim=True):

	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Matrix must be a colour image 3dimensional with 3rd dimension 3 colour channels')
	ideal_coords = select_random_point(mat)
	ideal = mat[ideal_coords]
	position = select_random_edge_point(mat)
	diffs = []
	coords = []

	h,w,ch = mat.shape
	diff = 100000

	tries=0
	max_tries = 1000

	while position != ideal_coords and tries <= max_tries:
		new_coords, diff = immediate_aversion_step(ideal, position,mat)
		diffs.append(diff)
		coords.append(new_coords)
		position = new_coords
		tries +=1

	if save_name is not None:
		save((diffs, coords), save_name)

	if gradient_anim and save_name is not None:
		slides = plot_anim_path(coords, h, w, ideal_coords, position, base=mat)
		sname = save_name + '_animation_slides'
		np.save(sname, slides)

	base = None
	if plot:
		base = plot_path(coords, h,w, base=mat)

	if return_base is True:
		return diffs, coords ,base
	return diffs, coords


def step_in_direction(mat, position, current_direction,step_size=1):
	dirh, dirw = current_direction
	curh, curw = position
	h,w,ch = mat.shape

	coords = (curh+dirh, curw+dirw)
	if check_proposed_points(coords, h,w) is True:
		return coords
	else:
		coords = random_walk_step(mat, position, step_size=step_size)
		return coords


def levy_flight_till_atop(mat, less_diff=0.1, alpha=50, save_name=None, plot=False, return_base=False, gradient_anim=False):
	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Matrix must be a colour image 3dimensional with 3rd dimension 3 colour channels')
	ideal_coords = select_random_point(mat)
	ideal = mat[ideal_coords]
	position = select_random_edge_point(mat)
	diffs = []
	coords = []

	h,w,ch = mat.shape
	diff = 100000

	tries=0
	max_tries = 1000

	curr_num_tries = 0
	current_direction = None

	while position != ideal_coords and tries <= max_tries:

		if curr_num_tries<=0 or current_direction is None:
			step_size = int(power_law_sample(alpha))
			curr_num_tries = step_size -1
			new_coords = random_walk_step(mat, position,step_size=1)
			diff = euclidean_distance(mat[new_coords], ideal)
			diffs.append(diff)
			coords.append(new_coords)
			dirx = position[0] - new_coords[0]
			diry = position[1] - new_coords[1]
			position = new_coords
			tries +=1
			current_direction = (dirx, diry)
		if curr_num_tries>0:
			new_coords = step_in_direction(mat, position,current_direction,step_size=1)
			diff = euclidean_distance(mat[new_coords], ideal)
			diffs.append(diff)
			coords.append(new_coords)
			position = new_coords
			curr_num_tries -=1
			tries+=1

	if save_name is not None:
		save((diffs, coords), save_name)

	if gradient_anim and save_name is not None:
		slides = plot_anim_path(coords, h, w, ideal_coords, position, base=mat)
		sname = save_name + 'animation_slides'
		np.save(sname, slides)

	base = None
	if plot:
		base = plot_path(coords, h,w, base=mat)
	print len(coords)
	print len(diffs)
	if return_base is True:
		return diffs, coords ,base
	return diffs, coords



def run_trial(N, step_fn, mat, less_diff=0.1, results_save=None, info=True):
	if len(mat.shape)!=3 and mat.shape[2]!=3:
		raise ValueError('Input matrix must be three dimensinoal and with three colour channels')
	
	h,w,ch = mat.shape
	all_coords = []
	all_diffs = []
	successes = []
	num_failures = 0
	num_successes = 0

	nums_till_success = []

	for i in xrange(N):
		diffs, coords = step_fn(mat, less_diff=less_diff,plot=False)
		n = len(diffs)
		assert n==len(coords), 'Something wrong here: differences and coordinates different lengths'
		if n<1000:
			num_successes+=1
		if n >=1000:
			num_failures+=1
		all_coords.append(coords)
		all_diffs.append(diffs)
		#if n<1000:
		assert len(coords) == len(diffs), 'Number of coordinates and differences is different'
		nums_till_success.append(len(coords))


	if results_save is not None:
		save_array(results_save+'_coords', all_coords)
		save_array(results_save+'_diffs',all_diffs)

	print "Number of failures = ", num_failures
	print "Number of success = " , num_successes

	nums_till_success = np.array(nums_till_success)

	return nums_till_success

def get_attractor_development_robustness(begin, end, save_base, save_after):
	mat = get_gradient_matrix(N=300, radius=5, save_name=save_base + '_matrix', save_after=save_after)
	for i in xrange(begin, end):
		mat = np.load(save_base +'_matrix' + '_' + str(i*10) + '.npy')
		random_nums = run_trial(10000, random_walk_till_atop, mat, less_diff=0.001)
		levy_nums = run_trial(10000, levy_flight_till_atop,mat, less_diff=0.001)
		gradient_nums = run_trial(10000, gradient_search_till_atop, mat, less_diff=0.001)
		np.save(save_base + '_random_' + str(i*10), random_nums)
		np.save(save_base + '_gradient_' + str(i*10), gradient_nums)
		np.save(save_base + '_levy_' + str(i*10), levy_nums)


	plot_attractor_devlopment_robustness(begin, end, save_name=save_base)

def select_single_point_in_radius(ch,cw,h,w):
	valid = False
	point  = (-10, -10) 
	while valid is False:
		direction = int(8*np.random.uniform(low=0, high=1))
		if direction == 0:
			point = (ch -1, cw-1)
		if direction == 1:
			point = (ch -1, cw)
		if direction == 2:
			point = (ch -1, cw + 1)
		if direction ==3:
			point = (ch, cw-1)
		if direction == 4:
			point = (-10, -10)
			valid = False
		if direction == 5:
			point = (ch, cw+1)
		if direction == 6:
			point = (ch+1, cw-1)
		if direction == 7:
			point = (ch+1, cw)
		if direction==8:
			point = (ch+1, cw+1)
		if direction > 8:
			return (ch-1, cw-1)
		if check_proposed_points(point, h,w) is True:
			print "is valid"
			valid = True
			return [point]


def generate_single_throughout_copy_dict(mat):
	h,w, ch = mat.shape
	copy_dict = {}
	for i in xrange(h):
		for j in xrange(w):
			copy_dict[(i,j)] = select_single_point_in_radius(i,j,h,w)
	return copy_dict

def generate_copy_dict(mat, num_to_copy, copy_radius, prob_no_learn=0):
	h,w,ch = mat.shape
	copy_dict = {}
	for i in xrange(h):
		for j in xrange(w):
			copy_list = []
			r = np.random.uniform(low=0, high=1)
			if r <= prob_no_learn:
				copy_list.append(-1)
			if r > prob_no_learn:
				for p in xrange(2*copy_radius+1):
					for q in xrange(2*copy_radius+1):
						xpoint = i - copy_radius + p
						ypoint = j - copy_radius+ q
						if xpoint >= 0 and xpoint < h:
							if ypoint >=0 and ypoint < w:
								rand = np.random.uniform(low=0, high=1)
								if rand <= (1/num_to_copy): 
									copy_list.append((xpoint, ypoint))
			copy_dict[(i,j)] = copy_list
	return copy_dict



def average_copy_points(mat, copy_list):
	N = len(copy_list)
	if N == 0:
		N =1 
	start_tup = (0,0,0)
	for tup in copy_list:
		start_tup = tuple_add(start_tup, mat[tup])
	return tuple_divide_scalar(start_tup, N)

def copy_dict_proportion_nonlearn(copy_dict):
	num = 0
	for k,v in copy_dict.iteritems():
		if v == [-1]:
			num+=1
	return num/len(copy_dict)

def copy_dict_average_step(mat, copy_dict, noise=None, noise_sigma=  0.1):
	if len(mat.shape)!=3 or mat.shape[2]!=3:
		raise ValueError('Matrix must be 2d colour image with 3 channels in format h,w,ch')

	h,w,ch = mat.shape
	new_mat = np.copy(mat)
	for i in xrange(h):
		for j in xrange(w):
			l = copy_dict[(i,j)]
			if l != [-1]: 
				if noise is not None:
					average = average_copy_points(mat, l)
					noise = tuple(np.random.normal(loc=noise, scale=noise_sigma, size=3))
					if i == 3 and j ==4:
						print average
						print tuple_add(average, noise)
					new_mat[i][j] = tuple_add(average, noise)

				if noise is None:
					new_mat[i][j] = average_copy_points(mat, l)
	return new_mat


def gradient_dimension_trials(final_N=1000,radius=2,trial_N=10000, save_per=10, dim_start=1, dim_end=20,save_name='dimensions/dimension_search'):
	results = []
	for i in range(dim_start,dim_end):
		dim_results = []
		for j in xrange(final_N//save_per):
			mat = get_gradient_matrix(N=j*save_per,radius=radius,plot=False, dim=i)
			num_till_success = run_trial(trial_N, gradient_search_till_atop,mat,less_diff=0.0001)
			dim_results.append(num_till_success)

		dim_results = np.array(dim_results)
		results.append(dim_results)
	results = np.array(results)
	print results.shape
	if save_name is not None:
		np.save(save_name, results)
	return results

def gradient_dimension_fixed_maps(final_N=1000,radius=2,trial_N=10000, save_per=10, dim_start=1, dim_end=20,save_name='dimensions/dimension_search'):
	results = []
	maps = []
	for l in xrange(final_N//save_per):
		pass
	for i in range(dim_start,dim_end):
		dim_results = []
		for j in xrange(final_N//save_per):
			mat = get_gradient_matrix(N=j*save_per,radius=radius,plot=False, dim=i)
			num_till_success = run_trial(trial_N, gradient_search_till_atop,mat,less_diff=0.0001)
			dim_results.append(num_till_success)
		dim_results = np.array(dim_results)
		results.append(dim_results)

	results = np.array(results)
	print results.shape
	if save_name is not None:
		np.save(save_name, results)
	return results





if __name__ == '__main__':
	#plot_image_changes()
	#mat = get_gradient_matrix(N=50, radius=5,save_name=None)
	#mat = np.load('gradient_matrix.npy')
	#plt.imshow(mat)
	#plt.show()
	#np.save('base_matrix_50',mat)
	#mat = np.load('matrix_1.npy')
	#plt.imshow(mat)
	#plt.show()
	#mat = np.load('base_matrix_50.npy')
	#plt.imshow(mat)
	#plt.show()
	#diffs, coords, base = levy_flight_till_atop(mat, save_name='levy_flight_search_3',plot=True,return_base=True, gradient_anim=True)
	#np.save('levy_flight_base_proper', base)
	#diffs, coords, base = gradient_search_till_atop(mat,save_name='gradient_search_path_5', plot=True,return_base=True)
	#np.save('gradient_base_proper_5',base)
	#diffs, coords, base = random_walk_till_atop(mat, save_name='random_walk_search_3', plot=True,return_base=True, gradient_anim=True)
	#np.save('random_walk_base_proper', base)

	"""
	random_nums = run_trial(10000, random_walk_till_atop, mat, less_diff=0.0001)
	levy_nums = run_trial(10000, levy_flight_till_atop,mat, less_diff=0.0001)
	gradient_nums = run_trial(10000, gradient_search_till_atop, mat, less_diff=0.0001)
	np.save('trial_random_actual_child', random_nums)
	np.save('trial_gradient_actual_child', gradient_nums)
	np.save('trial_levy_actual_child', levy_nums)
	#print len(random_nums)
	#print len(gradient_nums)
	#print "mean random: ", np.mean(random_nums)
	#print "gradient nums: " , np.mean(gradient_nums)
	#print "random variance", np.var(random_nums)
	#print "gradient variance: ", np.var(gradient_nums)
	
	
	rands = np.load('trial_random_actual_child.npy')
	gradients = np.load('trial_gradient_actual_child.npy')
	levys = np.load('trial_levy_actual_child.npy')
	print "means"
	print np.mean(rands)
	print np.mean(gradients)
	print np.mean(levys)
	print "variances"
	print np.var(rands)
	print np.var(gradients)
	print np.var(levys)

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

	plot_random_gradient_levys(rands, gradients, levys)
	"""

	#random_base = np.load('random_walk_base.npy')
	#levy_base = np.load('levy_flight_base.npy')
	#gradient_base = np.load('gradient_base.npy')
	
	#plot_example_gradient_and_random(random_base, gradient_base)
	#plot_example_random_levy_gradient(random_base, levy_base, gradient_base, base=mat)
	#plt.imshow(random_base)
	#plt.show()

	#plot the changes for animation purposes
	#plot_image_changes(N=200, radius=5,plot_after=1000000, save_name='vocal_learning_development_8')
	#this is actually going to test the robustness of the method
	#for i in range(5,20):
	#	save_name = 'vocal_learning_radius_' + str(i)
	#	plot_image_changes(N=200, radius=i, plot_after=1000000, save_name=save_name)
	#	print "completed version: " + str(i)


	#for i in range(3,20):
	#	osstr = 'python animate_seals.py vocal_learning_radius_' + str(i)+'.npy'
	#	os.system(osstr)
	#	print "done!

	#plot_copy_dict_development(N=600, radius=2,num_to_copy=1, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_2_1_600_no_learn_995_proper',prob_no_learn = 0.005)
	#os.system('python animate_seals_backup.py random_learning/copy_dict_2_1_600_no_learn_995_proper.npy')

	"""
	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_01_proper',prob_no_learn = 0.1)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_01_proper.npy')

	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_02_proper',prob_no_learn = 0.2)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_02_proper.npy')


	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_03_proper',prob_no_learn = 0.3)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_03_proper.npy')


	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_04_proper',prob_no_learn = 0.4)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_04_proper.npy')


	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_05_proper',prob_no_learn = 0.5)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_05_proper.npy')


	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_06_proper',prob_no_learn = 0.6)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_06_proper.npy')


	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_07_proper',prob_no_learn = 0.7)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_07_proper.npy')


	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_08_proper',prob_no_learn = 0.8)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_08_proper.npy')


	plot_copy_dict_development(N=600, radius=3,num_to_copy=2, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_3_2_600_no_learn_09_proper',prob_no_learn = 0.9)
	os.system('python animate_seals_backup.py random_learning/copy_dict_3_2_600_no_learn_09_proper.npy')
	"""
	#this is for testing the random copyin gof just a few seals next!
	#plot_image_changes(N=800, radius=1, plot_after=100000,save_name='random_learning/radius_1_800_0.3')
	#plot_image_changes(N=300, radius=4, plot_after=100000,save_name='random_learning/radius_4_300_0.3')
	#plot_image_changes(N=300, radius=5, plot_after=100000,save_name='random_learning/radius_5_300_0.3')
	#plot_image_changes(N=300, radius=6, plot_after=100000,save_name='random_learning/radius_6_300_0.3')
	#os.system('python animate_seals_backup.py random_learning/radius_3_300_0.3.npy')
	#os.system('python animate_seals_backup.py random_learning/radius_4_300_0.3.npy')
	#os.system('python animate_seals_backup.py random_learning/radius_5_300_0.3.npy')
	#os.system('python animate_seals_backup.py random_learning/radius_1_800_0.3.npy')

	#plot one and two for longer
	#plot_image_changes(N=400, radius=2, plot_after=100000, save_name='vocal_learning_radius_2')
	#plot_image_changes(N=400, radius=1, plot_after=100000, save_name='vocal_learning_radius_1')
	#os.system('python animate_seals.py vocal_learning_radius_2.npy')
	#os.system('python animate_seals.py vocal_learning_radius_1.npy')

	#test it with the random swapping to see if it helps
	#plot_image_changes(N=400, radius=3, plot_after=100000, save_name='vocal_learning_swap_test', swap_number=200)
	#os.system('python animate_seals.py vocal_learning_swap_test.npy')

	#for i in xrange(1,20):
		#mat = get_gradient_matrix(N=i*10, radius=5,save_name='gradient_matrix')
	#mat = np.load('gradient_matrix.npy')
	#	sname = 'base_matrix_' + str(i*10)
		#np.save(sname,mat)
		#random_nums = run_trial(10000, random_walk_till_atop, mat, less_diff=0.1)
		#levy_nums = run_trial(10000, levy_flight_till_atop,mat, less_diff=0.1)
		#gradient_nums = run_trial(10000, gradient_search_till_atop, mat, less_diff=0.1)
		#np.save('actual_child_random_' + str(i*10), random_nums)
		#np.save('actual_child_gradient_' + str(i*10), gradient_nums)
		#np.save('actual_child_levy_' + str(i*10), levy_nums)

		#rands = np.load('trial_random_proper.npy')
		#gradients = np.load('trial_gradient_proper.npy')
		#levys = np.load('trial_levy_proper.npy')
		#print "means"
		#print np.mean(rands)
		#print np.mean(gradients)
		#print np.mean(levys)
		#print "variances"
		#print np.var(rands)
		#print np.var(gradients)
		#print np.var(levys)

		#print "t-test rands gradients"
		#t,prob = t_test(rands, gradients)
		#print t 
		#print prob
		#print "t-test rands levys"
		#t,prob = t_test(rands, levys)
		#print t
		#print prob
		#print "t-test gradients levys"
		#t,prob = t_test(gradients, levys)
		#print t
		#print prob

		#plot_random_gradient_levys(random_nums, gradient_nums, levy_nums)
	
		
	#plot_attractor_devlopment_robustness(1,20,'trial_')
	#get_attractor_development_robustness(1,30,'actual_child', 10)
	#plot_attractor_devlopment_robustness(1,30, 'actual_child_', plot_individuals=True)

	
	#for i in xrange(3):
	#	plot_image_changes(N=1000, radius=1, plot_after=1000000, save_name='small_r/largeN/_radius_1_' + str(i))
	#	plot_image_changes(N=1000, radius=2, plot_after=1000000, save_name='small_r/largeN/radius_2_' + str(i))
		#plot_image_changes(N=400, radius=3, plot_after=1000000, save_name='small_r/update1/radius_3_' + str(i))
		#plot_image_changes(N=400, radius=4, plot_after=1000000, save_name='small_r/update1/radius_4_' + str(i))
	#plot_image_changes(N=400, radius=2, plot_after=100000, save_name='vocal_learning_radius_2')
	#plot_image_changes(N=400, radius=1, plot_after=100000, save

	#osstr = 'python animate_seals.py vocal_learning_radius_' + str(i)+'.npy'
	#for i in xrange(3):
	##	print "in loop!"
	#	osstr = 'python animate_seals_backup.py small_r/largeN/_radius_1_' + str(i)+'.npy'
		#osstr2 = 'python animate_seals_backup.py small_r/largeN/radius_2_' + str(i)+'.npy'
		#osstr3 = 'python animate_seals_backup.py small_r/update1/radius_3_' + str(i)+'.npy'
		#osstr4 = 'python animate_seals_backup.py small_r/update1/radius_4_' + str(i)+'.npy'
		#os.system(osstr)
		#os.system(osstr2)
		#os.system(osstr3)
		#os.system(osstr4)

#try development with mask
#mask = create_mask(cross_diagonal_mask, (50,50,3))
#plot_image_changes(N=600, radius=1, plot_after=100000, save_name='masks/cross_diagonal_1_600', mask=mask)
#os.system('python animate_seals_backup.py masks/cross_diagonal_1_600.npy')
#mask = create_mask(circle_mask, (50,50,3))
#plot_image_changes(N=600, radius=1, plot_after=100000, save_name='masks/circle_1_600', mask=mask)
#os.system('python animate_seals_backup.py masks/circle_1_600.npy')

	"""
	for i in xrange(6):
		random_nums = run_trial_LOS(10000, random_walk_LOS, mat, less_diff=0.0001,LOS_radius = i+1)
		levy_nums = run_trial_LOS(10000, levy_flight_LOS,mat, less_diff=0.0001,LOS_radius = i+1)
		gradient_nums = run_trial_LOS(10000, gradient_search_LOS, mat, less_diff=0.0001,LOS_radius = i+1)
		np.save('LOS_strategy/trial_LOS_random_' + str(i+1), random_nums)
		np.save('LOS_strategy/trial_LOS_gradient_' + str(i+1), gradient_nums)
		np.save('LOS_strategy/trial_LOS_levy_' + str(i+1), levy_nums)
	
	#then the graph loop
	for i in xrange(6):
		rands = np.load('LOS_strategy/trial_LOS_random_' + str(i+1) +'.npy')
		gradients = np.load('LOS_strategy/trial_LOS_gradient_' + str(i+1) +'.npy')
		levys = np.load('LOS_strategy/trial_LOS_levy_' + str(i+1) +'.npy')
		plot_random_gradient_levys(rands, gradients, levys)
	"""

	#run this a cuopleof billion times hoping for the bes so who knows about this!?
	# see if/how this develops... would be interesting!
	#copy_dict = generate_single_throughout_copy_dict(mat)
	#plot_copy_dict_development(N=600, radius=2,num_to_copy=3, plot_after=10000, save_after=1, save_name='random_learning/copy_dict_single_N_2',copy_dict = copy_dict)
	#os.system('python animate_seals_backup.py random_learning/copy_dict_single_N_2.npy')

	"""
	rands = np.load('LOS_strategy/trial_random_actual_child.npy')
	gradients = np.load('LOS_strategy/trial_gradient_actual_child.npy')
	levys = np.load('LOS_strategy/trial_levy_actual_child.npy')
	print "means"
	print np.mean(rands)
	print np.mean(gradients)
	print np.mean(levys)
	print "variances"
	print np.var(rands)
	print np.var(gradients)
	print np.var(levys)

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

	plot_random_gradient_levys(rands, gradients, levys)
	"""
	#plot_gradient_results(dims, (1,3,5,7,9,11,13,15,17), 10)


	#plot_copy_dict_development(N=600, radius=2,num_to_copy=2, plot_after=10000, save_after=1, save_name='noise/2_2_600_2_proper_noise_0_03_proper_shift', noise=0, noise_sigma=0.03)
	#os.system('python animate_seals_backup.py noise/2_2_600_2_proper_noise_0_03_proper_shift.npy')


	#plot_image_changes(N=600, radius=2, plot_after=100000, save_name='noise/full_average_600_2_0115',noise_sigma=0.115)
	#os.system('python animate_seals_backup.py noise/full_average_600_2_0115.npy')
	

	"""
	plot_image_changes(N=600,size=(30,30), radius=2, plot_after=100000, save_name='small/30x30_600_r_2')
	os.system('python animate_seals_backup.py small/30x30_600_r_2.npy')
	plot_image_changes(N=600,size=(20,20), radius=2, plot_after=100000, save_name='small/20x20_600_r_2')
	os.system('python animate_seals_backup.py small/20x20_600_r_2.npy')
	plot_image_changes(N=600,size=(10,10), radius=2, plot_after=100000, save_name='small/10x10_600_r_2')
	os.system('python animate_seals_backup.py small/10x10_600_r_2.npy')
	plot_image_changes(N=600,size=(5,5), radius=2, plot_after=100000, save_name='small/5x5_600_r_2')
	os.system('python animate_seals_backup.py small/5x5_600_r_2.npy')
	plot_image_changes(N=600,size=(3,3), radius=2, plot_after=100000, save_name='small/3x3_600_r_2')
	os.system('python animate_seals_backup.py small/3x3_600_r_2.npy')
	plot_image_changes(N=600,size=(3,3), radius=1, plot_after=100000, save_name='small/3x3_600_r_1')
	os.system('python animate_seals_backup.py small/3x3_600_r_1.npy')
	"""

	#plot_image_changes(N=600,size=(20,20), radius=2, plot_after=100000, constant_coords=(1,1),save_name='small/20x20_600_r_1_11_fixed_2')
	#os.system('python animate_seals_backup.py small/20x20_600_r_1_11_fixed_2.npy')

	#plot_image_changes(N=1500,size=(50,50), radius=2, plot_after=100000, constant_coords=None,constant_coords_frac=0.001,save_name='constant/50x50_1500_0001_2')
	#os.system('python animate_seals_backup.py constant/50x50_1500_0001_2.npy')

#
	#plot_image_changes(N=600,size=(100,100), radius=3, plot_after=100000, save_name='small/100x100_600_r_3')
	#os.system('python animate_seals_backup.py small/100x100_600_r_3.npy')
	#plot_image_changes(N=600,size=(200,200), radius=3, plot_after=100000, save_name='small/200x200_600_r_3')
	#os.system('python animate_seals_backup.py small/200x200_600_r_3.npy')
	#plot_image_changes(N=600,size=(500,500), radius=3, plot_after=100000, save_name='small/500x500_600_r_3')
	#os.system('python animate_seals_backup.py small/500x500_600_r_3.npy')
	#plot_image_changes(N=600,size=(1000,1000), radius=3, plot_after=100000, save_name='small/1000x1000_600_r_3')
	#os.system('python animate_seals_backup.py small/1000x1000_600_r_3.npy')
	