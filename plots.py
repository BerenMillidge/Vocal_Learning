# Plotting code file for Vocal learning papers
# Author: Beren Millidge
# Date: Summer 2018


import numpy as np 
import matplotlib.pyplot as plt

def plot_image_changes(N=150, size=(50,50), radius=5, plot_after=5, multiplier=0, save_after=1, save_name=None, swap_number = None, mask=None,dim=3, noise_sigma = None,constant_coords=None, constant_coords_frac=None):
	orig_mat = create_random_colour_matrix(size[0], size[1])
	save_list = []
	print "updating with radius: " + str(radius)
	if save_name is not None:
		save_list.append(orig_mat)

	constant_value = None
	if constant_coords_frac is not None:
		constant_coords = choose_constant_coords(orig_mat, constant_coords_frac)

	if constant_coords is not None:
		if isinstance(constant_coords, list):
			constant_value = []
			for coord in constant_coords:
				print coord
				constant_value.append(orig_mat[coord])
		else:
			constant_value = orig_mat[constant_coords]
	for i in xrange(N):
		orig_mat = matrix_average_step(orig_mat, radius,random_multiplier=multiplier,dim=dim, noise_sigma = noise_sigma,constant_coords=constant_coords, constant_value=constant_value)
		if swap_number is not None:
			orig_mat = randomise_swap_points(orig_mat, swap_number)
		if mask is not None:
			orig_mat = apply_mask_matrix(orig_mat, mask)
		print "plot: ", i
		if save_name is not None and i % save_after ==0:
			save_list.append(orig_mat)
		#if i % plot_after ==0:
			#plt.imshow(orig_mat)
			#plt.xticks([])
			#plt.yticks([])
			#plt.show()

	if save_name is not None:
		save_list = np.array(save_list)
		np.save(save_name, save_list)
	return orig_mat


def plot_path(coords, height, width,plot=True, base=None):
	if base is None:
		base = np.zeros((height,width))
	for i in xrange(len(coords)):
		x,y = coords[i]
		base[x][y] = 255.
	if plot:
		plt.imshow(base)
		plt.xticks([])
		plt.yticks([])
		plt.show()
	return base


def plot_anim_path(coords, height, width, ideal, position, base=None, flicker_parent=True, init_num=10):
	if base is None:
		base = np.zeros((height, width))
	slides = []
	for j in xrange(len(coords)):
		new_base = np.copy(base)
		x,y = coords[j]
		#print x, y
		parent_save = new_base[ideal]
		new_base[ideal] = (255, 255, 255)
		new_base[x][y] = (255, 255,255)
		if flicker_parent and j%2==0:
			new_base[ideal] = parent_save
		slides.append(new_base)

	slides = np.array(slides)
	print slides.shape
	return slides

def plot_example_gradient_and_random(random_base, gradient_base):
	fig = plt.figure()
	plt.title('Example Random and Gradient following paths through the colony')
	ax1 = fig.add_subplot(121)
	plt.imshow(random_base)
	plt.xticks([])
	plt.yticks([])
	ax2 = fig.add_subplot(122)
	plt.imshow(gradient_base)
	plt.xticks([])
	plt.yticks([])

	fig.tight_layout()
	plt.show()


def plot_example_random_levy_gradient(random_base, levy_base, gradient_base, base=None):
	fig = plt.figure()

	if base is not None:
		random_base = position_base_on_path(random_base, base)
		levy_base = position_base_on_path(levy_base, base)
		gradient_base = position_base_on_path(gradient_base, base)

	ax1 = plt.subplot(131)
	plt.imshow(random_base)
	ax1.set_title('Random walk path')
	plt.xticks([])
	plt.yticks([])

	ax2 = plt.subplot(132)
	plt.imshow(levy_base)
	ax2.set_title('Levy Flight path')
	plt.xticks([])
	plt.yticks([])

	ax3 = plt.subplot(133)
	plt.imshow(gradient_base)
	ax3.set_title('Gradient path')
	plt.xticks([])
	plt.yticks([])

	plt.subplots_adjust(wspace=0, hspace=0)
	fig.tight_layout()
	plt.show()


def plot_random_vs_gradient(randoms, gradients):
	rand_mu = np.mean(randoms)
	rand_var = np.var(randoms)
	gradient_mu = np.mean(gradients)
	gradient_var = np.var(gradients)
	fig  = plt.figure()

	plt.bar(rand_mu, label='Mean number of steps using a random walk')
	plt.bar(gradient_mu, label='Mean number of steps using a gradient search')
	fig.xlabel('Random walk or gradient search')
	fig.ylabel('Mean number of steps to reach target')
	plt.legend()
	fig.tight_layout()
	plt.show()

def plot_random_gradient_levys(randoms, gradients, levys):
	rand_mu = np.mean(randoms)
	gradient_mu = np.mean(gradients)
	levy_mu = np.mean(levys)
	rand_stderr = np.sqrt(np.var(randoms))/np.sqrt(len(randoms))
	gradient_stderr = np.sqrt(np.var(gradients))/np.sqrt(len(gradients))
	levy_stderr = np.sqrt(np.var(levys))/np.sqrt(len(levys))

	print "supposed standard deviations"
	print rand_stderr
	print gradient_stderr
	print levy_stderr

	print "confidence intervals"
	print scipy.stats.norm.interval(0.95, loc=rand_mu, scale=np.sqrt(np.var(randoms))/np.sqrt(len(randoms)))
	print scipy.stats.norm.interval(0.95, loc=gradient_mu, scale=np.sqrt(np.var(gradients))/np.sqrt(len(randoms)))
	print scipy.stats.norm.interval(0.95, loc=levy_mu, scale=np.sqrt(np.var(levys))/np.sqrt(len(randoms)))

	print len(randoms)
	labels = [r'$Random$', r'$L\acute{e}vy$',r'$Gradient$']
	errors = [rand_stderr, gradient_stderr, levy_stderr]

	pos = [1,2,3]
	res = [rand_mu, levy_mu, gradient_mu]
	fig, ax = plt.subplots()
	ax.bar(pos, res, width=0.6, yerr=errors, tick_label=labels, align='center', alpha=0.8, ecolor='black', capsize=10)
	ax.set_xlabel('Search Strategy')
	ax.set_ylabel('Mean number of steps to reach infant')
	ax.yaxis.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_attractor_devlopment_robustness(begin,end, save_name, show=True, plot_individuals=False):
	rand_means = []
	gradient_means = []
	levy_means = []
	for i in xrange(begin, end):
		rname = save_name + 'random_' + str(i*10) + '.npy'
		gname = save_name + 'gradient_' + str(i*10) + '.npy'
		lname = save_name + 'levy_' + str(i*10) + '.npy'
		randoms = np.load(rname)
		gradients = np.load(gname)
		levys = np.load(lname)
		rand_means.append(np.mean(randoms))
		gradient_means.append(np.mean(gradients))
		levy_means.append(np.mean(levys))
		if plot_individuals:
			t_test_results(randoms, gradients, levys)
			plot_random_gradient_levys(randoms, gradients, levys)

	rand_means = np.array(rand_means)
	gradient_means = np.array(gradient_means)
	levy_means = np.array(levy_means)

	nums = np.array(xrange(begin, end))*10

	fig = plt.figure()
	plt.plot(nums, rand_means, label='Random Walk')
	plt.plot(nums, gradient_means, label='Gradient Walk')
	plt.plot(nums, levy_means, label=r'L$\acute{e}$vy Flight')
	plt.xlabel('Number of epochs of nest development')
	plt.ylabel('Mean number of steps to reach infant')
	plt.legend()
	plt.tight_layout()
	if show:
		plt.show()
	return fig



def plot_copy_dict_development(N=150, radius=5,num_to_copy=3, plot_after=5, save_after=1, save_name=None,copy_dict = None, prob_no_learn=0,noise=None, noise_sigma=0.1):
	orig_mat = create_random_colour_matrix(50,50)
	if copy_dict is None:
		print "generating copy dict"
		copy_dict = generate_copy_dict(orig_mat, num_to_copy, radius,prob_no_learn = prob_no_learn)
	save_list = []
	print "updating with radius: " + str(radius)
	if save_name is not None:
		save_list.append(orig_mat)

	for i in xrange(N):
		print copy_dict_proportion_nonlearn(copy_dict)
		orig_mat = copy_dict_average_step(orig_mat, copy_dict, noise = noise, noise_sigma = noise_sigma)
		print "plot: ", i
		if save_name is not None and i % save_after ==0:
			save_list.append(orig_mat)

	if save_name is not None:
		save_list = np.array(save_list)
		np.save(save_name, save_list)
	return orig_mat


def plot_gradient_results(results, dims,save_per):
	sh = results.shape
	fig = plt.figure()
	xs = range(0, sh[1]*save_per,save_per)
	for dim in dims:
		if dim <0 or dim >=sh[0]:
			raise ValueError('Dimension not included in the results matrix')
		res = results[dim]
		means = []
		for i in xrange(len(res)):
			means.append(np.mean(res[i]))
		means = np.array(means)
		plt.plot(xs, means, label='Dimension ' + str(dim))
	plt.xlabel('Epochs of colony development')
	plt.ylabel('Mean steps till success')
	plt.title('Effect of call dimensionality on gradient search effectiveness')
	plt.legend()
	plt.tight_layout()
	plt.show()
	return fig