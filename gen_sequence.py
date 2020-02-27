from texture_transformations import *
#from pylab import imshow, show, get_cmap
from numpy import *
import cv2
#import matplotlib.pyplot as plt

#Generate a random grayscale image on which the transformations will be applied
random_image_gray = np.random.random((512, 512))   # Randomly generated image rendered in grayscale)
max_kernel_size = 10
blend_params_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rotation_params_list = [0,-45, 45, -90, 90, -135, 135, -180, 180, -225, 225, -270, 270, -315, 315,-360, 360]
translation_params_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
affine_params_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
persp_params_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

transforms_list = ['blur', 'translate', 'rotate', 'affine', 'perspective']		# sequence of possible image transformations
max_len = 4                                                            		        # maximum length of generated program sequence            

"""
	Generate a random sequence of transforms to be applied to an image
	Can genrate a dataset of n! numpy arrays denoting 
"""

class Program():
	def __init__(self, transforms):
		self.transforms = transforms

	def apply_program(self, image):
		out = image
		for trans in self.transforms:
			out = trans.apply(out)
		return out

	def get_type_sequence(self):
		return tuple([ trans.name for trans in self.transforms])


class Transform():
	def apply(self, image):
		raise UnImplementedError


class Blur(Transform):
	def __init__(self,kernel_width):
		self.ks = (kernel_width,kernel_width)
		self.name = 'blur'

	def apply(self, image):
		return im_blur(image, self.ks)


def sample_rand_prog(transforms_list):

	prog = []

	random_program_length = random.randint(1, max_len+1)
	for ele in range(random_program_length):
		trans_type = random.choice(transforms_list)

		if trans_type == 'blur':
			kernel_width = random.choice(range(1, max_kernel_size))
			#trans = Blur(kernel_width)
			trans = (trans_type,(kernel_width,kernel_width))

		elif trans_type == 'rotate':
			angle = random.choice(rotation_params_list)
			trans = (trans_type,angle)

		elif trans_type=='translate':
			translation_matrix = np.float32([[1,0,random.randint(10,100)],[0,1,random.randint(10,100)]])
			trans = (trans_type,translation_matrix)

		elif trans_type == 'blend':
		 	weights = random.choice(blend_params_list)
		 	trans = (trans_type,(weights, 1-weights))

		elif trans_type == 'perspective':
			pts1_psp = np.float32([[0,0],[random.randint(50,100),0],[0,random.randint(50,100)],[random.randint(50,100),random.randint(50,100)]])
			pts2_psp = np.float32([[0,0],[random.randint(50,100),0],[0,random.randint(50,100)],[random.randint(50,100),random.randint(50,100)]])
			pts_psp = (pts1_psp, pts2_psp)
			trans = (trans_type, pts_psp)
			
		elif trans_type == 'affine':
			pts1_aff = np.float32([[random.randint(20,25),random.randint(20,25)],[random.randint(50,55),random.randint(20,25)],[random.randint(35,40),random.randint(80,90)]])
			pts2_aff = np.float32([[random.randint(20,25),random.randint(20,25)],[random.randint(50,55),random.randint(20,25)],[random.randint(35,40),random.randint(80,90)]])
			pts_aff = (pts1_aff,pts2_aff)
			trans = (trans_type,pts_aff)
		
		assert ele == len(prog), "Program sequence not matching number of times loop was run"
		prog.append(trans)

	return prog

def apply_transform(im, prog):

	if prog[0] == 'blur':
		out_im = im_blur(im,prog[1])

	# if prog[0] == 'blend':
	# 	out_im = im_blend(im, im, prog[1])

	elif prog[0] == 'affine':
		out_im = affine_transform(im, prog[1])

	elif prog[0] == 'rotate':
		out_im = im_rotate(im, prog[1])

	elif prog[0] == 'translate':
		out_im = im_translate(im, prog[1])

	elif prog[0] == 'perspective':
		out_im = perspective_transform(im, prog[1]) 

	return out_im


def generate_textures(program):
	
	textures = []
	for ele in program:
		textures.append(ele[0])
	return textures

def generate_transforms(program):
	
	instruction_list = []

	for ele in program:
		instruction_list.append(ele[1])
	return instruction_list

def apply_program(input_im,program):

	output_im = input_im
	transforms_list = []
	for prog in program:
		output_im = apply_transform(output_im,prog)
		transforms_list.append(prog[0])
	return (cv2.resize(output_im, (64,64), interpolation = cv2.INTER_AREA),transforms_list)  #resized to smaller images for training

def word_to_int(word):
	if word == '<start>':
		return 0
	if word == '<end>':
		return 6
	if word == 'blur':
		return 1
	elif word == 'rotate':
		return 2
	elif word == 'translate':
		return 3
	elif word == 'affine':
		return 4
	elif word == 'perspective':
		return 5
	# elif word == 'blend':
	# 	return 6
	elif word == 'pad':
		return 7
	return -1

def int_to_word(int_):
	if int_ == 0:
		return '<start>'
	if int_ == 6:
		return '<end>'
	if int_ == 7:
		return 'pad'
	if int_ == 1:
		return 'blur'
	elif int_ == 2:
		return 'rotate'
	elif int_ == 3:
		return 'translate'
	elif int_ == 4:
		return 'affine'
	elif int_ == 5:
		return 'perspective'
	# elif int_ == 6:
	# 	return 'blend'
	return 'NaN'
#print(sample_rand_prog(transforms_list))
#program = sample_rand_prog(transforms_list) 
#print(len(program))						
#print(generate_transforms(program))										#serves as a checker to know the set of program instructions we started with																						
#print(apply_program(random_image_gray,program))


## Rendering a random set of image transformations

#print(apply_program(random_image_gray,program)[0])														#should be the same as the output from the generate_instruction_list()
#print(apply_program(random_image_gray,program)[1])														#the set of program instructions

#out_im = apply_program(random_image_gray,program)[0]

#imshow(out_im, cmap=get_cmap("gray"), interpolation='nearest')   				#Display output image
#show()
#plt.imsave('textures/texture1.png', out_im, cmap="gray")

## Generating training data
batch_size = 5000

def generate_training_data(batch_size):
	training_data = []
	for ele in range(batch_size):
		prog = sample_rand_prog(transforms_list)
		prog_im_sequence = apply_program(random_image_gray,prog)
		training_data.append(prog_im_sequence)
	return training_data

training_data = generate_training_data(batch_size)
textures_train = generate_textures(training_data)
transforms_train = generate_transforms(training_data)
#print(training_data)
# for i in range(len(textures_train)):
# 	plt.imsave('sequences/seq10_%i.png'%i, textures_train[i], cmap="gray")
#print(transforms_train)       											
#get the transforms used for each texture






