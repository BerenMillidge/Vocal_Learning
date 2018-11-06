# Masks code file for Vocal learning papers
# Author: Beren Millidge
# Date: Summer 2018



import numpy as np


def diagonal_mask(x,y, diff=15):
	if x + diff > y and x -diff < y:
		return (1,1,1)
	else: 
		return (0,0,0)

def half_diagonal_mask(x,y,diff=5):
	if x+diff > y:
		return (1,1,1)
	else:
		return (0,0,0)

def circle_mask(x,y, ch=25,cw=25, r=15):
	if euclidean_distance((x,y), (ch, cw)) <=r:
		return (1,1,1)
	else:
		return (0,0,0)

def cross_diagonal_mask(x,y, diff=10):
	if x + diff > y and x -diff < y:
		return (1,1,1)
	if y + x > 50-diff and x + y < 50 + diff:
		return (1,1,1)

	else: 
		return (0,0,0)



def create_mask(f, shape):
	mat = np.zeros(shape)
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			mat[i][j] = f(i,j)
	return mat

#mask = create_mask(cross_diagonal_mask, (50,50,3))
#print mask
#plt.imshow(mask)
#plt.show()

def create_random_mask(shape, multiplier):
	if len(shape)!=3:
		raise ValueError('Shape must be three dimensional for colour image')
	height,width,channels = shape
	return multiplier * np.random.randn(height, width, channels)


def apply_mask_matrix(mat, mask):
	if mat.shape != mask.shape:
		raise ValueError('Matrix and mask must be same shape')
	h,w,ch = mat.shape
	for i in xrange(h):
		for j in xrange(w):
			mat[i][j] = mat[i][j] * mask[i][j]
	return mat

def apply_mask_matrix_addition(mat, mask):
	if mat.shape != mask.shape:
		raise ValueError('Matrix and mask must be same shape')
	h,w,ch = mat.shape
	for i in xrange(h):
		for j in xrange(w):
			mat[i][j] = mat[i][j] + mask[i][j]
	return mat

def combine_masks(mask_list):
	for i in xrange(len(mask_list)-1):
		if mask_list[i].shape != mask_list[i+1].shape:
			raise ValueError('All masks must have the same shape')
	comb_mask = np.zeros(mask_list[0].shape)
	for mask in mask_list:
		h,w = mask.shape
		for i in xrange(h):
			for j in xrange(w):
				if mask[i][j] >0:
					comb_mask[i][j] = 1.

	return comb_mask