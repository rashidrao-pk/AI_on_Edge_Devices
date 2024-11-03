import numpy as np
import tensorflow as tf
import pandas as pd

import platform
import keras
import sys,os
import time,datetime
from tqdm.auto import tqdm
import argparse

from AI_on_Edge_Devices.codes.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test_model',  type=bool,  default=True,     help='Set to True to train the model.')
parser.add_argument('--dataset',      type=str,   default='cifar10',help='Camera to be used.')
parser.add_argument('--model_name',   type=str,   default='custom', help='Model to be used.')
parser.add_argument('--top_preds',   type=int,   default=3, help='Model to be used.')
parser.add_argument('--test_data_type', type=str,   default='loaded',    help='') 
parser.add_argument('--testdevice',     type=str,   default='raspberry',   help='Path to save/saved results.')
parser.add_argument('--results_path',     type=str,   default='results',   help='Path to save/saved results.')
parser.add_argument('--verbose',      type=bool,    help='show progress.')
args = parser.parse_args()

top_preds = 3

start_time = time.time()
codes_path = 'AI_on_Edge_Devices/codes'
args.results_path = os.path.join(codes_path,'results')

model_path = os.path.join(codes_path, 'models/cifar_converted_model.tflite')

file_name  = 'AI_on_Edge_Devices/docs//devices//raspberry_results.txt'

print(f'Loading data from {args.test_data_type}')
if args.test_data_type=='file':
	testX = np.load('AI_on_Edge_Devices/codes/saved_test_x.npy')
	testY = np.load('AI_on_Edge_Devices/codes/saved_test_y.npy')
	# class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
else:
	dataset_manager = DatasetManager(dataset_name=args.dataset,preprocess_data=True)
	trainX, trainY, testX, testY = dataset_manager.load_dataset(binary2multi=False)
	class_names = dataset_manager.get_class_names()
    #   from tensorflow.keras.datasets import cifar10 
    #   (_, _), (testX, testY) = cifar10.load_data()
	
if len(testY.shape)==2:
      testY = testY.flatten()


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Function to log and print messages
def log_and_print(message, file):
    print(message)
    file.write(message + '\n')
    
def write_environment_and_results(file_name='AI_on_Edge_Devices/docs/devices/full_results.txt'):
    with open(file_name, "w") as file:
        def log_and_write(message):
            print(message)
            file.write(message + '\n')
	# Write environment info
        log_and_write(f'{"Python version:":<25} {sys.version}')
        log_and_write(f'{"TensorFlow version:":<25} {tf.__version__}')
        log_and_write(f'{"Keras version:":<25} {keras.__version__}')
        log_and_write(f'{"NumPy version:":<25} {np.__version__}')
        log_and_write(f'{"System architecture:":<25} {platform.architecture()}')
        log_and_write(f'{"Machine type:":<25} {platform.machine()}')
        log_and_write(f'{"Processor:":<25} {platform.processor()}')
        log_and_write(f'{"Platform:":<25} {platform.system()}, {platform.version()}')
        log_and_write(f'{"Release:":<25} {platform.release()}')
        log_and_write(f'{"Version:":<25} {platform.version()}')

        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            log_and_write("TensorFlow GPU Devices:")
            for device in gpu_devices:
                log_and_write(f"{device}")
        else:
            log_and_write("No GPU detected by TensorFlow")

        log_and_write('-' * 120)
        log_and_write(f"{'Dataset':<25} {args.dataset}")
        log_and_write(f"{'Model Type':<25} {args.model_name}")
        log_and_write(f"{'Test Device':<25} {args.testdevice}")
        current_time = datetime.datetime.now().strftime("%H:%M:%S | %d-%m-%Y ")
        log_and_write(f'{"Experiments Made":<25} {current_time}')
        log_and_write('-' * 120)

        return file

data_csv = []
predictions,predY_tfl = [],[]
test_verbose = False
print('-'*120)

file = write_environment_and_results(file_name = file_name)
print('-'*120)
with open(file_name, 'a') as file:
	log_and_print(f'{"Test data INFO":<25} {testX.shape} {testY.shape}', file)
	log_and_print(f'{"Images INFO":<25} {type(testX[0])} {testX[0].dtype}', file)
	log_and_print(f'{"Using Model":<25} {model_path}', file)
	log_and_print(f'{"Storing INFO to":<25} {file_name}', file)
print('-'*120)
#for idd,(img,lbl) in enumerate(tqdm(zip(testX,testY))):
for idd,(img,lbl) in enumerate(zip(tqdm(testX),testY)):
    start_epoch_time = time.time()
    processed_img = np.expand_dims(img, axis=0).astype(np.float32)  #float16
    interpreter.set_tensor(input_index, processed_img)
    interpreter.invoke()
    prediction_output = interpreter.get_tensor(output_index)
    prediction = np.argmax(prediction_output)

    predY_tfl.append(prediction)
    data_csv.append({
		'id': idd,
		'test_lbl': lbl,
		'prediction_lbl': prediction,
		'true_class': class_names[lbl],
        'prediction_class': class_names[prediction],
        'probability': float(np.max(prediction_output) * 100),
        'prediction_time': f"{time.time() - start_epoch_time:.2f}s",
        'testdevice': args.testdevice,
        'model_type': args.model_name,
    })

df = pd.DataFrame(data_csv)

df.to_csv(f'{args.results_path}/{args.dataset}_{args.model_name}_test_results_{args.testdevice}.csv', index = False)

predY_tfl = np.array(predY_tfl, dtype=np.uint8)

predictions_results = np.concatenate((predY_tfl, testY))

np.save(f'predY_tfl_raspberry', predictions_results)

evaluation_tfl_raspi = Evaluation(y_true=testY,y_pred=predY_tfl, args=args)

results_tfl_raspi = evaluation_tfl_raspi.evaluate_model()

with open(file_name, 'a') as file:
    for perf_measures, perf_values in results_tfl_raspi.items():
        message = f'{perf_measures:<25} : {perf_values:<25.7}'
        print(message)
        file.write(message + '\n')
        
    elapsed_total_time = time.time() - start_time
    elapsed_total_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total_time))
    total_time_message = f'Total Time Taken: {elapsed_total_time_str}'
    print(total_time_message)
    file.write('-' * 120 + '\n')
    file.write(total_time_message + '\n')
    file.write('-' * 120 + '\n')

