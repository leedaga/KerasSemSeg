import sys, skvideo.io, json, base64, os
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import models
from keras.models import load_model
file = sys.argv[-1]

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")


video = skvideo.io.vread(file)

answer_key = {}

opt = {}

opt['weights_file'] = os.path.join(os.path.dirname(__file__), 'mobilnet_fcnn2.h5')
opt['nb_classes'] = 3
opt['input_shape'] = (600, 800, 3)
opt['presence_weight'] = 50.0
#model_file = os.path.join(os.path.dirname(__file__), 'mobilenet_fcnn_mdl.h5')
#print('loading from', opt['weights_file'])

model = models.create_model(opt)
model.load_weights(opt['weights_file'])

#from keras.applications import mobilenet
#model = load_model(model_file, custom_objects={
#                   'relu6': mobilenet.relu6,
#                   'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    
# Frame numbering starts at 1
frame = 1

for rgb_frame in video:

    #segment image based on model
    pred = model.predict(rgb_frame[None, :, :, :])[0]

    #car channel
    car_ch = pred[:,:,2]
    binary_car_result = np.where(car_ch>0,1,0).astype('uint8')

    #road channel
    road_ch = pred[:,:,1]
    binary_road_result = np.where(road_ch>0,1,0).astype('uint8')

    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]

    # Increment frame
    frame+=1

# Print output in proper json format
print (json.dumps(answer_key))