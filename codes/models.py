##########################################################################################################################################################
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,GlobalAveragePooling2D

##########################################################################################################################################################
###########                      CUSTOM CNN
##########################################################################################################################################################
class Models():
    def __init__(self,args=None,input_shape=(32, 32, 3),num_classes = 10):
        self.input_shape    = input_shape
        self.args           = args
        self.model_type     = args.model_name
        self.dataset        = args.dataset
        self.num_classes    = num_classes

    def get_model(self):
        if self.model_type=='custom':
            return self.custom_model_cifar10()
        elif self.model_type=='vgg16':
            return self.get_vgg16()
        elif self.model_type=='resnet50':
            return self.get_resnet50()
        else:
            raise ValueError("Unsupported model type. Choose 'ResNet' or 'VGG'.")
    ##########################################################################################################################################################
    def custom_model_cifar10(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        # compile model
        opt = SGD(lr=self.args.lr, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    ##########################################################################################################################################################
    def get_resnet50(self):
        model = ResNet50(weights=None, include_top=False, input_shape=self.input_shape)
        model = Sequential([
            model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    ##########################################################################################################################################################
    def get_vgg16(self):
        model = VGG16(weights=None, include_top=False, input_shape=self.input_shape)
        model = Sequential([
            model,
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    