# Strategies code file for Vocal learning papers
# Author: Beren Millidge
# Date: Summer 2018


import numpy as np 
import matplotlib.pyplot as plt


def line_of_sight_euclidean(current_coords, ideal_coords, LOS_radius):
	if euclidean_distance(current_coords, ideal_coords) <= LOS_radius:
		return True
	return False

def line_of_sight_coords(current_coords, ideal_coords, LOS_radius):
	cw, ch = current_coords
	iw,ih = ideal_coords
	if iw <= cw + LOS_radius and iw >= cw - LOS_radius:
		if ih <= ch + LOS_radius and ih >= ch - LOS_radius:
			return True
	return False


def gradient_search_LOS(mat, less_diff=0.01, save_name=None, plot=False,return_base=False, gradient_anim=True, LOS_radius=3, LOS_func=line_of_sight_euclidean):

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

	while LOS_func(position, ideal_coords, LOS_radius) is False and tries <= max_tries:
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

def random_walk_LOS(mat, less_diff=0.1, step_size=1,save_name=None, plot=False, plot_animation=False,return_base=False, gradient_anim=False,LOS_radius=3, LOS_func=line_of_sight_euclidean):
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

	while LOS_func(position, ideal_coords, LOS_radius) is False and tries <= max_tries:

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


def levy_flight_LOS(mat, less_diff=0.1, alpha=50, save_name=None, plot=False, return_base=False, gradient_anim=False,LOS_radius=3, LOS_func=line_of_sight_euclidean):
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

	while LOS_func(position, ideal_coords, LOS_radius) is False and tries <= max_tries:
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


def run_trial_LOS(N, step_fn, mat, less_diff=0.1, results_save=None, info=True, LOS_radius=3):
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
		diffs, coords = step_fn(mat, less_diff=less_diff,plot=False,LOS_radius=LOS_radius)
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