# Sequential Modelling in Data-Driven Approach for Structural Health Monitoring by Recurrent Convolutional Neural Networks

**Conference:** 20th World Conference on Non-Destructive Testing    
**Location & Date:** South Korea, Seoul - June 2020      
**Co-authors:** Ewald V., Goby X., Groves R.M. & Benedictus R.             
**Labarotory:** TU Delft Aerospace NDT Lab

## Usage Instruction

In order to make use of this project all you need mainly be concerned with is the main.py
Python script. In it you shall (hopefully) find yourself a more than sufficient amount of
documentation in order to understand and be able to make use of it! 

## Dev Progress Log-Journal, Data Characteristics & Background Information

### Notes:

The CNN subsection/layers is/are used for performing feature on the
input data.
   
The LSTM subsection/layers is/are used for providing sequential
prediction capability. In other words, the LSTM subsection/layers are
used in order to support the ability of providing a sequence of images
as input data as opposed to only a single image. Alternatively, the LSTM
subsection/layers can be used for generating a sequence in response to an
input image provided.

Sequence Prediction Example: Given the input of multiple time steps of
a damped sine wave, predict the next few time steps of the sequence.    

The 1st hidden layer of an LSTM must define the number of inputs which
is to be expected e.g. "the shape of the input layer". The shape of the
data which his input to an LSTM (i.e. @ the 1st LSTM hidden layer) must
be 3D and specify the following:
- The number of samples. 1 sample = 1 sequence
- The number of time-steps. This is the # of past observations for a
  feature e.g. lag variables.
- The number of features


### Log-Journal:


###### December 14 Saturday 2019   

    To-Do(s):
    
    Learn about RNNs and LSTMs
    
    Key Take-Aways:
    
    1) Typically, recurrent layers such as LSTM accept 
       input of the shape (batch_size, input_sequence_length, features). The following 
       examples explain the two common use cases.
    2) If the return_sequences and return_state parameters are provided with False then an 
       LSTM layer will return an output tensor with shape 
       (batch_size, num_lstm_units)=(None, num_lstm_units). 
       Providing both parameters with True instead will return the an output tensor with its 
       complete shape.
_____________________________________________________________________
###### December 21 Saturday 2019  

    To-Do(s):
    
    Fix repository after issue of data having gone "missing" :(, i.e. the
    images and continue with RNN and LSTM research
_____________________________________________________________________
###### December 25 Wednesday 2019    

    To-Do(s):
    
    Implement LSTM (make use of the work I have already done for my LSTM models for algorithmic trading)
 _____________________________________________________________________
###### December 31 Tuesday 2019     

    To-Do(s):
    
    Code image data loading functionality. Have a look at some examples of "data generators" coded from
    scratch which are also being used with Keras. Try to avoid using Keras ImageDataGenerator unless it
    out that this is truly unnecessary.
    
    Key Take-Aways:
    
    1) The batch normalization performed by the BatchNormalization function in keras is the one proposed by 
       Ioffe & Szegedy, 2015 which is applicable for fully-connected and convolutional layers only9.
_____________________________________________________________________
###### January 2 Thursday 2020     

    To-Do(s):
    
    Finally start reading up on the  ConvLSTM Precipitation paper. Then research more on ConvLSTM.
    
    Key Take-Aways:
    
    1) In essence, precipitation nowcasting is a spatiotemporal sequence forecasting problem
       with the sequence of past radar maps as input and the sequence of a fixed number (usually larger
       than 1) of future radar maps as output.
       
    2) By stacking multiple ConvLSTM layers and forming an encoding-forecasting structure, we can build an
       end-to-end trainable model for precipitation nowcasting. 
_____________________________________________________________________
###### January 5 Saturday 2020    

    To-Do(s):
    
    Attempt to implement ConvLSTM, investigate Keras Conv2DLSTM layer
_____________________________________________________________________
###### January 12 Saturday 2020     

    To-Do(s):
    
    Learn how to use Keras Functional API.
    
    Key Take-Aways:
    
    1) Dense, Activation, Reshaape, Conv2D and LSTM are examples of layers which inherit/are derived from the
       abstract Layer class!
    2) Layer objects are callable because they have a __call__ method2. The __call__ method accepts 
       a tensor or a list/tuple of tensors and returns tensor or a list/tuple of tensors.
    3) Even though we Input lies within keras.layers, Input is not actually a Layer object. 
       Input is a function. Calling Input returns a tensor, as we have seen above. 
       Input function calls the InputLayer class, which is indeed a subclass of Layer. 
       InputLayer instantiates a tensor which is returned to us as the output of the Input function.
_____________________________________________________________________
###### January 19 Saturday 2020

    To-Do(s):
    
    Reimplement ConvLSTM w/o using Keras Conv2DLSTM layer but instead using functional API
_____________________________________________________________________
###### February 11 Tuesday & 12 Wednesday 2020

    To-Do(s):

    Meet up with Vincent to discuss the issue of whethe it is seq2seq preds which I am doing or not.
    TURNS out nope, it is not seq2seq that we are doing but instead the following:
    I have got to take each image and divide them into smaller frames (along the width of each images, i.e. div of 4101 pixels).
    Then I assign the class label of the OG raw undivided large image (of width 4101 pxs) to each of the frames obtained from the process mentioned above.
    Then I feed this and use the grid square moving and pred of direction of movement e.g. to continue working from here on then get back to VBoss
_____________________________________________________________________
###### February 24 Monday 2020

    To-Do(s):

    Go ahead and implement the functionality to permit the slicing up of a single given image into multiple smaller sized images AKA "frames"
    of equal width(= 4101/num of frames). Then make sure that I also implement the functionality that allows for each frame generated to
    inherit a class label of its own which is the same of that of the OG raw large unsliced image. Then go ahead and start perform the necessary verification
    validation of this process and its output for the use case of feeding the output of ImageDataSource() in load_image_data.py into the conv(2d)lstm nn
    via the init input layer! This'll take time to get right b/c of the nuisance of shape/dimensions compatibility.
_____________________________________________________________________
###### February 26 Wednesday 2020

    Notes:
    Reduce 4100 dts AKA img px width down to 4100 which has divisors 1, 2, 4, 5, 10, 20, 25, 41, 50, 82, 100, 164, 205, 410, 820, 1025, 2050
    4100[ms] = 4.1[s] therefore
    Let frame pxs width = 25 AKA 1 frame represents 25 [ms] dts
    4100 / 25 = 164 so there are 164 total frames
    1000 / 25 = 40
    1 img is 4100[ms]=4.1[s] long and so I want to be sampling each image at 40 frames per second so as to get 164 frames per img

    To-Do(s):


_____________________________________________________________________
###### March 5 Thursday 2020

    Notes:
    - "We can define a CNN LSTM model in Keras by first defining the CNN layer or layers, wrapping
    them in a TimeDistributed layer and then defining the LSTM and output layers. We have
    two ways to define the model that are equivalent and only differ as a matter of taste. You can
    define the CNN model first, then add it to the LSTM model by wrapping the entire sequence of
    CNN layers in a TimeDistributed layer. An alternate, and perhaps easier to read, approach is to wrap each layer in the CNN model
    in a TimeDistributed layer when adding it to the main model. The benefit of this second approach is that
    all of the layers appear in the model summary"

    - In keras, fit() is much similar to sklearn's fit method, where you pass array of features as x values and target as y values.
    You pass your whole dataset at once in fit method. Also, use it if you can load whole data into your memory (small dataset).

    - In fit_generator(), you don't pass the x and y directly, instead they come from a generator. As it is written in keras
    documentation, generator is used when you want to avoid duplicate data when using multiprocessing. This is for
    practical purpose, when you have large dataset.

    To-Do(s):

    - Focus on modularizing the project more.

    - Migrate all CNN, LSTM & MLP models created &/or used so far into their seperate own functions so as to simplify the procedure
    of building/instantiating available models. Have the models be defined within their own respective functions which are then built
    and returned as the output of the functions!

    - Work on settings.py for ease of access to frequently used parameter settings

    - Create a main.py file within which combinations of various CNN, LSTM and MLP models and the TimeDistributed layer wrapper
    can be tested out!

    - Add custom functions for saving trained models either completely (arch + weights + optimizer state) or only
    partially (e.g. arch)
  
_____________________________________________________________________
###### March 6th Monday 2020
    
    Notes:
    
    - Incorporated more LSTM, CNN and CONVLSTM models and also did some refactoring
    
    - Implemented the ability to plot a visualisation of the architecture of a network/model. 
    
    - Finally managed to "get data into and back out from the CONVLSTM model/network" succesfully and so without
    experiencing any issues with tensor data representations, i.e. shape
    
_____________________________________________________________________
###### March 27th Monday 2020

    Notes:
    
    - Seems like having selected the Mark Directory as | Sources Route option for some of the 
    modules was nothing more than duck-tape quick fixes of importation issues which I would run into during development. 
    The project has now been structured in the most appropriate/conventional Python way and it should now fix all the importation 
    related issues which Vincent was experiencing. 
    
    -  Tidied up this README.md file
    
    



_____________________________________________________________________
## Network Input Layer and Input Data Shapes Notes:

A Conv2D layer requires four dimensions, not three: (batch_size, width, height, channels)

And the TimeDistributed will require an additional dimension: (batch_size, frames, width, height, channels)

So in order to work with a TimeDistributed + Conv2D layer I need to have 5 dimensions!

E.g. input shape to pass to a TimeDistributed+Conv2D layer: (frames, height, width, channels)
model.add(TimeDistributed(Dense(....), input_shape = (frames, width, height, channels))

(batch_size, time(_steps), width, height, channels) <=> (batch_size, frames, width, height, channels)

"If you have a black and white video, for instance, then you have only one channel.
In a numpy array, you can simply reshape it. Suppose you have an array x_train with shape (10,86,28,28).
Then: x_train=x_train.reshape((10,86,28,28,1))" - source: https://stackoverflow.com/questions/47470385/use-kerastensorflow-to-build-a-conv2dlstm-model
(10,86,28,28,1) = (batch_size, frames, width, height, channels)

The # of frames (AKA # of time_steps) specifies the number of frames (time_steps) which are processed before an output is returned/provided.

E.g. 5dim data shape: (sample, time, width, length, channel)


## Key terms/topics/concepts:

PZT (Effect)

Lamb wave shm

Lamb wave based structural health Monitoring

Guided wave SHM

lamb wave shm signal processing with machine learning

## Relevant research papers & posts:
- RNN with Attention for Genre Classification: https://pdfs.semanticscholar.org/bff3/eaf5d8ebb6e613ae0146158b2b5346ee7323.pdf
- Music Genre Recognition: http://deepsound.io/music_genre_recognition.html
- Using CNNs and RNNs for Music Genre Recognition: https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af

## Description of data.csv:
Data recorded via PZT   
Type: Time Series   
Rows, Columns = 600, 801    

The 1st row of data along with all its columns, i.e. [0,0:801], corresponds with time series #1   
The 2nd row of data along with all its columns, i.e. [1,0:801], corresponds with time series #2   
...     
The 600th row of data along with all its columns, i.e. [599,0:801], corresponds with time series #600   
So in essence:  
Sample #1 <=> Time series #1 <=> Row #1     
Sample #2 <=> Time series #2 <=> Row #2     
...     
Sample #600 <=> Time series #600 <=> Row #600     

## Description of label.csv:
Type: Time Series   
Rows, Columns: 600, 1   

Label #1 <=> Target of time series #1 <=> Row #1    
Label #2 <=> Target of time series #2 <=> Row #2    
...     
Label #600 <=> Target of time series #600 <=> Row #600  

