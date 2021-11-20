import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model

# Get The Data
(x_train , _) , (x_test , _ ) = mnist.load_data()

# Reshape The Data
x_train = x_train.reshape(x_train.shape[0] , x_train.shape[1] , x_train.shape[2] , 1)
x_test = x_test.reshape(x_test.shape[0] , x_test.shape[1] , x_test.shape[2] , 1)

# Standarize The Data
x_train = x_train / 255
x_test  = x_test / 255

# Function to display The Data

def display_data(data , height, width, title):
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.imshow(data[i].reshape((height,width)))
        plt.gray()        
    plt.suptitle(title)
    
# Our Convolution AutoEncoder    
def Conv_AutoEncoder() :
    #Input Layer
    input_layer = Input(shape=(28, 28, 1), name="input_layer")
    # Encoder    
    x = Conv2D(16, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2) , strides=(2,2))(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2) , strides=(2,2) , name = 'code_layer')(x)
    # Code
    code = x
    # Decoder
    x = UpSampling2D((2, 2))(code)
    x = Conv2DTranspose(8, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2) )(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu')(x)
    # Output Layer
    output_layer = Conv2DTranspose(1, (3, 3), activation='sigmoid')(x)
    
    conv_Autoencoder = Model(input_layer , output_layer)
    
    return conv_Autoencoder
    
# Compiling The Model

Convolution_AutoEncoder = Conv_AutoEncoder()
Convolution_AutoEncoder.compile(optimizer='adam', loss='mse')
Convolution_AutoEncoder.summary()

# Training The Model
Convolution_AutoEncoder.fit(x_train , x_train , epochs=10 , batch_size=32 , shuffle=True , validation_data=(x_test , x_test))

# get The Decoded Data
decoded_data = Convolution_AutoEncoder.predict(x_test)

Encoder = Model(inputs = Convolution_AutoEncoder.input , outputs = Convolution_AutoEncoder.get_layer("code_layer").output)

# get The Encoded Data
encoded_data = Encoder.predict(x_test)

# display The Result

display_data(x_test , height = 28 , width = 28 , title = 'Original Data')
display_data(encoded_data , height = 5 , width = 5*8 , title = 'Encoded Data')
display_data(decoded_data , height = 28 , width = 28 , title = 'Decoded Data')    