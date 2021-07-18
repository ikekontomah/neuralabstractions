import cv2
import numpy as np
from numpy import *
from scipy import *
import matplotlib.pyplot as plt

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
	return cv2.addWeighted(im1, weights[0], im2, weights[1], 0)

def im_translate(im,translation_matrix):
	rows,cols, _ = im.shape					# Used for real world 3D images
	dst = cv2.warpAffine(im, translation_matrix, (cols,rows))
	return dst

def im_rotate(im, degree_of_rotation):
	rows,cols, _ = im.shape					# Used for real world 3D images        
	rotation_matrix = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),degree_of_rotation,1)
	dst = cv2.warpAffine(im,rotation_matrix,(cols,rows))
	return dst

def affine_transform(im,points_aff):	
	rows,cols, _ = im.shape					# Used for real world 3D images
	affine_transform_matrix = cv2.getAffineTransform(points_aff[0],points_aff[1])
	dst = cv2.warpAffine(im,affine_transform_matrix,(cols,rows))
	return dst

def perspective_transform(im,points_psp):
	rows,cols, _ = im.shape					# Used for real world 3D images 
	perspective_transform_matrix = cv2.getPerspectiveTransform(points_psp[0],points_psp[1])
	dst = cv2.warpPerspective(im,perspective_transform_matrix,(256,256))
	return dst

## Composite image transformations

## Do an image translation and rotate it
def translate_rotate(im):	
	im = im_translate(im,translation_matrix)
	rows,cols, _ = im.shape					# Used for real world 3D images 	
	rotation_matrix = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),degree_of_rotation,1)
	dst = cv2.warpAffine(im,rotation_matrix,(cols,rows))
	return dst

def affine_perspective(im):
	im = affine_transform(im,points_aff) 
	rows,cols, _ = im.shape					# Used for real world 3D images 
	points_psp = (pts1_psp,pts2_psp)
	perspective_transform_matrix = cv2.getPerspectiveTransform(points_psp[0],points_psp[1])
	dst = cv2.warpPerspective(im,perspective_transform_matrix,(256,256))
	return dst


def display_blur(im):
	cv2.imshow('blurred image',im_blur(im,kernel_size)) 

def display_blend(im1,im2):
	cv2.imshow('Blended images',im_blend(im1,im2,weights))

def display_translation(im):
	cv2.imshow('Translated', im_translate(im,translation_matrix))

def display_rotation(im):
	cv2.imshow('Rotated', im_rotate(im,degree_of_rotation))

def display_affine_transformation(im):
	cv2.imshow('Affine transform', affine_transform(im,points_aff))

def display_perspective_transformation(im):
	cv2.imshow('Perspective transform', perspective_transform(im,points_psp))

def display_translate_rotate(im):
	cv2.imshow("Translation Rotation", translate_rotate(im))

def display_affine_perspective(im):
	cv2.imshow("Affine perspective",affine_perspective(im))

im1 = cv2.imread('test_images/green.jpg')
im2 = cv2.imread('test_images/icy.jpg')
im3 = cv2.imread('test_images/clouds.jpg')
im4 = cv2.imread('test_images/lake.jpg')

w1,h1,_ = im1.shape	
w2,h2,_ = im2.shape
w3,h3,_ = im3.shape
w4,h4,_ = im4.shape
# print(w1,h1)
# print(w2,h2)
# print(w3,h3)
# print(w4,h4)
im1 = cv2.resize(im1,(256,256))
im2 = cv2.resize(im2,(256,256))
im3 = cv2.resize(im3,(256,256))
im4 = cv2.resize(im4,(256,256))

w1,h1,_ = im1.shape	
w2,h2,_ = im2.shape
w3,h3,_ = im3.shape
w4,h4,_ = im4.shape
# print(w1,h1)
# print(w2,h2)
# print(w3,h3)
# print(w4,h4)

p1 = np.float32([[random.randint(20,25),random.randint(20,25)],[random.randint(50,55),random.randint(20,25)],[random.randint(35,40),random.randint(80,90)]])
p2 = np.float32([[random.randint(20,25),random.randint(20,25)],[random.randint(50,55),random.randint(20,25)],[random.randint(35,40),random.randint(80,90)]])
p3 = np.float32([[0,0],[random.randint(50,100),0],[0,random.randint(50,100)],[random.randint(50,100),random.randint(50,100)]])
p4 = np.float32([[0,0],[random.randint(50,100),0],[0,random.randint(50,100)],[random.randint(50,100),random.randint(50,100)]])
#Different blur transforms on colored images

# blur1 = cv2.blur(im1,(5,5))
# cv2.imwrite("transforms_rgb/green_blur.png",blur1)

# blur2 = cv2.GaussianBlur(im2,(5,5),0)
# cv2.imwrite("transforms_rgb/icy_blur.png",blur2)

# blur3 = cv2.medianBlur(im3,5)
# cv2.imwrite("transforms_rgb/clouds_blur.png",blur3)

# blur4 = cv2.bilateralFilter(im4,10,80,80)
# cv2.imwrite("transforms_rgb/lake_blur.png",blur4)

# #Different Blend Transforms

# blend1 = im_blend(im1,im2,(0.5,0.5))
# cv2.imwrite("transforms_rgb/green_icy.png",blend1)

# blend2 = im_blend(im2,im3, (0.5,0.5))
# cv2.imwrite("transforms_rgb/icy_clouds.png",blend2)

# blend3 = im_blend(im3,im4, (0.5,0.5))
# cv2.imwrite("transforms_rgb/clouds_lake.png",blend3)

# blend4 = im_blend(im1,im4, (0.5,0.5))
# cv2.imwrite("transforms_rgb/green_lake.png",blend4)

# #Different translation transforms on colored images

# trans1 = im_translate(im1, np.float32([[1,0,20],[0,1,20]]))
# cv2.imwrite("transforms_rgb/green_trans.png",trans1)

# trans2 = im_translate(im2, np.float32([[1,0,40],[0,1,40]]))
# cv2.imwrite("transforms_rgb/icy_trans.png",trans2)

# trans3 = im_translate(im3, np.float32([[1,0,60],[0,1,60]]))
# cv2.imwrite("transforms_rgb/clouds_trans.png",trans3)

# trans4 = im_translate(im4, np.float32([[1,0,80],[0,1,80]]))
# cv2.imwrite("transforms_rgb/lake_trans.png",trans4)

# #Different rotation transforms on colored images

# rot1 = im_rotate(im1, 90)
# cv2.imwrite("transforms_rgb/green_rot.png",rot1)

# rot2 = im_rotate(im2, 120)
# cv2.imwrite("transforms_rgb/icy_rot.png",rot2)

# rot3 = im_rotate(im3, 180)
# cv2.imwrite("transforms_rgb/clouds_rot.png",rot3)

# rot4 = im_rotate(im4, 270)
# cv2.imwrite("transforms_rgb/lake_rot.png",rot4)

# #Different affine transforms on colored images

# aff1 = affine_transform(im1, (p1,p2))
# cv2.imwrite("transforms_rgb/green_aff.png",aff1)

# aff2 = affine_transform(im2, (p1,p2))
# cv2.imwrite("transforms_rgb/icy_aff.png", aff2)

# aff3 = affine_transform(im3, (p1,p2))
# cv2.imwrite("transforms_rgb/clouds_aff.png", aff3)

# aff4 = affine_transform(im4, (p1,p2))
# cv2.imwrite("transforms_rgb/lake_aff.png", aff4)

#Different perspective transforms on colored images

psp1 = perspective_transform(im1, (p3,p4))
cv2.imwrite("transforms_rgb/green_psp.png", psp1)

psp2 = perspective_transform(im2, (p3,p4))
cv2.imwrite("transforms_rgb/icy_psp.png", psp2)

psp3 = perspective_transform(im3, (p3,p4))
cv2.imwrite("transforms_rgb/clouds_psp.png", psp3)

psp4 = perspective_transform(im4, (p3,p4))
cv2.imwrite("transforms_rgb/lake_psp.png", psp4)
#display_blur(im1)
#display_blend(im1,im2)
#display_translation(im1)
#display_rotation(im1)
#display_affine_transformation(im1)
#display_perspective_transformation(im1)
#display_translate_rotate(im1)
#display_affine_perspective(im1)
#cv2.waitKey(0) 