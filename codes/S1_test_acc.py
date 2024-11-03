import numpy as np
import tensorflow as tf

import cv2
import pandas as pd

import platform
import keras
import sys
import time
from tqdm.auto import tqdm
import argparse

# from
# from AI_on_Edge_Devices.codes.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_model',  type=bool,  default=True,     help='Set to True to train the model.')
parser.add_argument('--dataset',      type=str,   default='cifar10',help='Camera to be used.')
parser.add_argument('--model_name',   type=str,   default='custom', help='Model to be used.')

parser.add_argument('--test_data_type', type=str,   default='loaded',    help='Path to save/saved results.') # file
parser.add_argument('--testdevice',     type=str,   default='raspberry',   help='Path to save/saved results.')

parser.add_argument('--verbose',      type=bool,    help='show progress.')
args = parser.parse_args()

class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
top_preds = 3
model_path = 'AI_on_Edge_Devices/codes/models/cifar_converted_model.tflite'
print(f'Loading data from {args.test_data_type}')
if args.test_data_type=='file':
	testX = np.load('AI_on_Edge_Devices/codes/saved_test_x.npy')
	testY = np.load('AI_on_Edge_Devices/codes/saved_test_y.npy')
else:
      from tensorflow.keras.datasets import cifar10 
      (_, _), (testX, testY) = cifar10.load_data()


print('Test data INFO : ',testX.shape, testY.shape)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def get_environment_info(file_name='rasp_environment_info.txt'):
    import keras
    with open(file_name, "w") as file:
        def log_and_print(message):
            print(message)  
            file.write(message + '\n')  
        log_and_print(f'{"Python version:":<25} {sys.version}')
        log_and_print(f'{"TensorFlow version:":<25} {tf.__version__}')
        log_and_print(f'{"Keras version:":<25} {keras.__version__}')
        log_and_print(f'{"NumPy version:":<25} {np.__version__}')
        log_and_print(f'{"System architecture:":<25} {platform.architecture()}')
        log_and_print(f'{"Machine type:":<25} {platform.machine()}')
        log_and_print(f'{"Processor:":<25} {platform.processor()}')
        log_and_print(f'{"Platform:":<25} {platform.system()}, {platform.version()}')
        log_and_print(f'{"Release:":<25} {platform.release()}')
        log_and_print(f'{"Version:":<25} {platform.version()}')

        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            file.write("TensorFlow GPU Devices:\n")
            for device in gpu_devices:
                file.write(f"{device}\n")
        else:
            file.write("No GPU detected by TensorFlow\n")


def preprocess_frame(frame,img_size=(32,32)):
    frame = cv2.resize(frame, img_size)             
    frame = frame / 255.0                           
    frame = np.expand_dims(frame, axis=0).astype('float32')  # float32'        
    return frame
data_csv = []
predictions,predY_tfl = [],[]
test_verbose = False
print('-'*120)
get_environment_info(file_name='AI_on_Edge_Devices/docs//devices//raspberry_environment_info.txt')
print('-'*120)
#for idd,(img,lbl) in enumerate(tqdm(zip(testX,testY))):
for idd,(img,lbl) in enumerate(zip(tqdm(testX),testY)):
	epoch_start_time = time.time()
	if len(lbl.shape)!=0:
		lbl = lbl[0]
	data = {}
	# print(img.shape)
	
	processed_img  = np.expand_dims(img, axis=0).astype('float32')  # float32' 
	# print(processed_img.shape)
	# processed_img = preprocess_frame(img)
	interpreter.set_tensor(input_index, processed_img)
	interpreter.invoke()

	top_k_values, top_k_indices = tf.nn.top_k(interpreter.get_tensor(output_index), top_preds)
	
	topk = top_k_indices[0].numpy()
	topv = top_k_values[0].numpy()

	prediction = np.argmax(interpreter.get_tensor(output_index))
	if test_verbose:
		# print(lbl,prediction, lbl.shape, prediction.shape)
		if class_names[lbl] == class_names[prediction]:
			print(f'{"CORRECT":<10}: {idd:<10}/{testX.shape[0]:<10} : {class_names[lbl]:<20}, - ,{class_names[prediction]:<20} - {topv[0]}')
		else:
			print(f'{"WRONG":<10}: {idd}/{testX.shape[0]} : {class_names[lbl]:<20}, - ,{class_names[prediction]:<20} - {topv}')

	prediction_cifar = {class_name : 0.0 for class_name in class_names}
	prediction_val = []
	
	for tk,tv,rr in zip(topk,topv,range(0,top_preds,1)):
		prediction_val.append(f'{class_names[topk[rr]]}:{topv[rr]*100:0.2f}')
	predictions.append(prediction_val)
	epoch_end_time = time.time()
	elapsed_time = epoch_end_time - epoch_start_time
	elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
	predY_tfl.append(prediction)
	data = {'id':idd,
		'lbl':lbl,
		'prediction':prediction,
		'lbl_str':class_names[lbl],
		'prediction_str':class_names[prediction],
		'prediction_time':elapsed_time_str,
		'testdevice':'raspberry',
		'model_type':'TFL',
		}
	data_csv.append(data)
df = pd.DataFrame(data_csv)

df.to_csv(f'{args.dataset}_{args.model_name}_test_results_{args.testdevice}.csv', index = False)
predY_tfl = np.array(predY_tfl)
np.save(f'predY_tfl_raspberry', predY_tfl)
# print(predictions[0].shape, testY.shape)


