import cv2
import numpy as np
from numpy import *
from scipy import *
#import matplotlib.pyplot as plt

# Hyperparameters for image transformations
kernel_size = (10,10) 												#parameters for blurring an image done by im_blur
weights = (0.5,0.5)  												#parameters for blending an image done by im_blend
translation_matrix = np.float32([[1,0,50],[0,1,50]]) 				#parameters for translating an image done by im_translate
degree_of_rotation = 90                               				#parameter for rotating an image by a given angle done by im_rotate

pts1_aff = np.float32([[20,20],[50,20],[35,80]])
pts2_aff = np.float32([[20,20],[50,20],[35,80]])
points_aff = (pts1_aff,pts2_aff)									#parameters for doing an affine transform done by affine_transform

pts1_psp = np.float32([[20,20],[50,20],[35,80],[60,90]])
pts2_psp = np.float32([[0,0],[300,0],[0,300],[300,300]])			#parameters for doing a perspective transform done by perspective_transform
points_psp = (pts1_psp,pts2_psp)


def im_blur(im,kernel_size):
	return cv2.blur(im,kernel_size)

def im_blend(im1,im2,weights):
	
	im1_width,im1_height = im1.shape			
	im2_width,im2_height = im2.shape
	return cv2.addWeighted(im1, weights[0], cv2.resize(im2,(im1_width,im1_height)), weights[1], 0)

def im_translate(im,translation_matrix):
	
	rows,cols = im.shape						
	dst = cv2.warpAffine(im, translation_matrix, (cols,rows))
	return dst

def im_rotate(im, degree_of_rotation):
	        
	rows,cols = im.shape						
	rotation_matrix = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),degree_of_rotation,1)
	dst = cv2.warpAffine(im,rotation_matrix,(cols,rows))
	return dst

def affine_transform(im,points_aff):
	
	rows,cols = im.shape						
	affine_transform_matrix = cv2.getAffineTransform(points_aff[0],points_aff[1])
	#print(affine_transform_matrix)
	dst = cv2.warpAffine(im,affine_transform_matrix,(cols,rows))
	return dst

def perspective_transform(im,points_psp):

	rows,cols = im.shape						
	perspective_transform_matrix = cv2.getPerspectiveTransform(points_psp[0],points_psp[1])
	#print(perspective_transform_matrix)
	dst = cv2.warpPerspective(im,perspective_transform_matrix,(512,512))
	return dst

## Composite image transformations

## Do an image translation and rotate it
def translate_rotate(im):
	
	im = im_translate(im,translation_matrix) 
	rows,cols = im.shape						# Used for synthetic 2D images		
	rotation_matrix = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),degree_of_rotation,1)
	dst = cv2.warpAffine(im,rotation_matrix,(cols,rows))
	return dst

def affine_perspective(im):
	im = affine_transform(im,points_aff) 
	rows,cols = im.shape						# Used for synthetic 2D images
	points_psp = (pts1_psp,pts2_psp)
	perspective_transform_matrix = cv2.getPerspectiveTransform(points_psp[0],points_psp[1])
	dst = cv2.warpPerspective(im,perspective_transform_matrix,(512, 512))
	return dst


# Show what each of the transforms does for a couple of different parameters
# on 2-D grayscale images

#im = random.random((512,512))   # Randomly generated image rendered in grayscale
# blurs = []
# for i in range(1,11):
# 	blurs.append(im_blur(im,(i,i)))
# for i in range(len(blurs)):
# 	plt.imsave('transforms/blur%i.png'%i, blurs[i], cmap="gray")

# blend_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# blends = []
# for i in blend_params:
# 	blends.append(im_blend(im,im,(i,1-i)))
# for i in range(len(blends)):
# 	plt.imsave('transforms/blend%i.png'%i, blends[i], cmap="gray")

# rotation_params = [45, 60, 90, 135,180, 225, 270, 300, 315, 360]
# rotations = []
# for i in rotation_params:
# 	rotations.append(im_rotate(im,i))
# for i in range(len(rotations)):
# 	plt.imsave('transforms/rotation%i.png'%i, rotations[i], cmap="gray")

# translation_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# translations = []
# for i in translation_params:
# 	translations.append(im_translate(im,np.float32([[1,0,i],[0,1,i]])))
# for i in range(len(translations)):
# 	plt.imsave('transforms/translation%i.png'%i, translations[i], cmap="gray")

# p1 = np.float32([[random.randint(20,25),random.randint(20,25)],[random.randint(50,55),random.randint(20,25)],[random.randint(35,40),random.randint(80,90)]])
# p2 = np.float32([[random.randint(20,25),random.randint(20,25)],[random.randint(50,55),random.randint(20,25)],[random.randint(35,40),random.randint(80,90)]])
# affine_params = (p1,p2)
# affines = []
# for i in range(10):
# 	affines.append(affine_transform(im,affine_params))
# for i in range(len(affines)):
# 	plt.imsave('transforms/affine%i.png'%i, affines[i], cmap="gray")

#p3 = np.float32([[0,0],[random.randint(50,100),0],[0,random.randint(50,100)],[random.randint(50,100),random.randint(50,100)]])
#p4 = np.float32([[0,0],[random.randint(50,100),0],[0,random.randint(50,100)],[random.randint(50,100),random.randint(50,100)]])
#persp_params = (p3,p4)
#persps = []
#for i in range(10):
#	persps.append(perspective_transform(im,persp_params))
#	print(persp_params)
#for i in range(len(persps)):
#	plt.imsave('transforms/persps%i.png'%i, persps[i], cmap="gray")
