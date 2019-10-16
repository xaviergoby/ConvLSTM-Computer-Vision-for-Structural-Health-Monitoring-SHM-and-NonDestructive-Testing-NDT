# Data Characteristics & Background Information

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










index   label   0   1   2   3   ... 600
0       0       .   .   .   .   ... .