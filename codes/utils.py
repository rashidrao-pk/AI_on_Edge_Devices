import cv2
import sys
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
##########################################################################################################################################################
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score,confusion_matrix)
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
##########################################################################################################################################################
###########                      CLASS : EVALUATION
##########################################################################################################################################################
class Evaluation():
    def __init__(self, dataset_name='cifar10',y_true=None,y_pred=None,average_type='macro', beta=0.5, zero_division=1,verbose=False,args=None):
            self.dataset_name  = dataset_name
            self.y_true        = y_true
            self.y_pred        = y_pred
            self.average_type  = average_type
            self.beta          = beta
            self.zero_division = zero_division
            self.verbose       = verbose
            self.args          = args
    def evaluate_model(self):
        y_true = self.y_true
        y_pred = self.y_pred
        # Suppress UndefinedMetricWarning warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)

            # Ensure y_true and y_pred are compatible in shape
            if y_true.ndim == 2:
                y_true = y_true.flatten()

            # Core evaluation metrics with zero_division parameter
            accuracy = accuracy_score  (y_true, y_pred)
            precision = precision_score(y_true, y_pred, average=self.average_type, zero_division=self.zero_division)
            recall = recall_score      (y_true, y_pred, average=self.average_type, zero_division=self.zero_division)
            f1 = f1_score              (y_true, y_pred, average=self.average_type, zero_division=self.zero_division)
            f_beta = fbeta_score       (y_true, y_pred, beta=self.beta, average=self.average_type, zero_division=self.zero_division)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

            # Additional metrics for multiclass evaluation
            cohen_kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            # Compile all metrics into a dictionary
            results = {
                'Accuracy': accuracy,
                'Balanced_Accuracy': balanced_accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'F_Beta': f_beta,
                'cohen_kappa': cohen_kappa, # Cohen Kappa Score
                'MCC': mcc, # Matthews Correlation Coefficient (MCC)
            }
        if self.verbose:
            # Print all metrics in a clear format
            for metric, value in results.items():
                print(f"{metric}:")
                print(value, "\n" if isinstance(value, str) else f"{value:.4f}")

        return results
    def plot_confusion_matrix(self,model_type='full_model',dpi=200, transparent=True,ext='png'):
        y_true = self.y_true
        y_pred = self.y_pred
        
        datamanager = DatasetManager()
        class_names = datamanager.get_class_names()
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create a heatmap
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.savefig(f'{self.args.path_results}/{self.args.dataset}_{self.args.model_name}_{model_type}_confmat_{self.args.testdevice}.{ext}',
                    bbox_inches='tight',transparent=transparent, dpi=dpi, )
        plt.show()
    def summarize_diagnostics(history):
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history['loss'], color='blue', label='train')
        plt.plot(history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history['accuracy'], color='blue', label='train')
        plt.plot(history['val_accuracy'], color='orange', label='test')
        # save plot to file
        filename = sys.argv[0].split('/')[-1]
        # plt.savefig(f'{self.args.}/{filename} + '_plot.png')
    # pyplot.close()
# scale pixels

##########################################################################################################################################################
###########                      CLASS : DatasetManager
##########################################################################################################################################################
class DatasetManager:
    def __init__(self, dataset_name='cifar10',preprocess_data=True):
            self.dataset_name    = dataset_name
            self.class_names     = self.get_class_names()
            self.preprocess_data = preprocess_data
            
    def get_class_names(self):
        if self.dataset_name == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            raise ValueError("Dataset not supported")

        
    # load train and test dataset
    def load_dataset(self,binary2multi=True,preprocess_data=True):
        if self.dataset_name=='cifar10':
            # load dataset
            (trainX, trainY), (testX, testY) = cifar10.load_data()
        if binary2multi:
            # one hot encode target values
            trainY = to_categorical(trainY)
            testY  = to_categorical(testY)
        else:
            if trainY.ndim == 2:
                trainY,testY = trainY.flatten(), testY.flatten()
        if self.preprocess_data:
            trainX, testX = self.prep_pixels(trainX, testX)
        
        return trainX, trainY, testX, testY
    @staticmethod
    def prep_pixels(train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm
##########################################################################################################################################################
###########                      CLASS Inference
##########################################################################################################################################################
class Inference():
    def __init__(self, dataset_name='cifar10',model=None):
            self.dataset_name    = dataset_name
            self.class_names     = self.get_class_names()
            self.img_size        = self.get_img_size()
            self.model           = model
#             self.preprocess_data = preprocess_data
        # Define function for preprocessing the frame to fit model input
    def get_img_size(self):
        if self.dataset_name=='cifar10':
            return (32, 32)
        else:
            raise ValueError("Dataset not supported")
    def get_class_names(self):
        if self.dataset_name == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            raise ValueError("Dataset not supported")
    def preprocess_frame(self,frame):
        frame = cv2.resize(frame, self.img_size)             
        frame = frame / 255.0                           
        frame = np.expand_dims(frame, axis=0)           
        return frame
    def predict_frame(self,frame):
        preprocessed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(preprocessed_frame)[0]  # Get prediction scores
        return predictions
    def get_caminfo(cap):
        # Fetch common camera properties
        camera_info = {
            "Backend": cap.get(cv2.CAP_PROP_BACKEND),
            "Frame Width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "Frame Height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "FPS": cap.get(cv2.CAP_PROP_FPS),
            "FourCC Code": cap.get(cv2.CAP_PROP_FOURCC),
            "Brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "Contrast": cap.get(cv2.CAP_PROP_CONTRAST),
            "Saturation": cap.get(cv2.CAP_PROP_SATURATION),
            "Hue": cap.get(cv2.CAP_PROP_HUE),
            "Exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "Gain": cap.get(cv2.CAP_PROP_GAIN),
            "Focus": cap.get(cv2.CAP_PROP_FOCUS),
        }

        # Display the information
        for prop, value in camera_info.items():
            print(f"{prop}: {value}")

        # Additional information (for specific backends like DirectShow, V4L, etc.)
        backend_name = {
            cv2.CAP_ANY: "Auto",
            cv2.CAP_V4L2: "Video4Linux2",
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation",
            cv2.CAP_FFMPEG: "FFMPEG",
            cv2.CAP_GSTREAMER: "GStreamer",
        }
        backend = int(cap.get(cv2.CAP_PROP_BACKEND))
        print(f"Backend used: {backend_name.get(backend, 'Unknown')}")
        # Release the camera when done
##########################################################################################################################################################
###########                      CLASS Inspection
##########################################################################################################################################################
class Inspection():
    def __init__(self,model=None):
            self.model    = model
    def model_stats(self,print_summary=True):
        model = self.model
        trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        model_size = total_params * 4 / (1024 ** 2)

        if print_summary:
            model.summary()
        print(f"{'Trainable Parameters':30} {trainable_params}")
        print(f"{'Non-Trainable Parameters':30} {non_trainable_params}")
        print(f"{'Total Parameters:':30} {total_params}")
        print(f"{'Model Size (MB)':30} {model_size:.2f}")

        # Approximate FLOPs
        flops = 0
        for layer in model.layers:
            if isinstance(layer, Conv2D):
                _, h, w, _ = layer.output_shape
                flops += h * w * layer.filters * (layer.kernel_size[0] * layer.kernel_size[1] * layer.input_shape[-1])
            elif isinstance(layer, Dense):
                flops += layer.input_shape[-1] * layer.units

        print(f"{'Approximate FLOPs':30} {flops / (10 ** 6):.2f} MFLOPs")
##########################################################################################################################################################