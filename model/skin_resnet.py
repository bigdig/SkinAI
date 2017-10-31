# encoding: UTF-8
""" a 2D resnet based on keras resnet50
"""
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K


class skin_resnet(object):
    	
	def __init__(self):
		base_model = ResNet50(include_top=False)
		# add a global spatial average pooling layer
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		# dropout
		# x = Dropout(0.2)(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		# and a logistic layer -- let's say we have 2 classes
		predictions = Dense(2, activation='softmax')(x)
		# this is the model we will train
		self.model = Model(inputs=base_model.input, outputs=predictions)
		self.model.load_weights("model_weight/resnet50_turnable_152_d2_3.h5") # 导入权重

	def predict(self,image):
    		return self.model.predict(image)


