# Log-Journal, Data Characteristics & Background Information

## Log-Journal:


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
# Key terms/topics/concepts:

PZT (Effect)

Lamb wave shm

Lamb wave based structural health Monitoring

Guided wave SHM

lamb wave shm signal processing with machine learning

# Relevant research papers & posts:
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

