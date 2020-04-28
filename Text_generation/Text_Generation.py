import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint



# loading the file
file = open(r"C:\Users\win10\Desktop\frank2.txt").read()


# tokenization
def tokenize_words(inputs):
    inputs = inputs.lower
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(inputs)
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return "".join(filtered)


processed_inputs = tokenize_words(file)


# Changing chars to integers
chars = sorted(list(set(processed_inputs)))
chars_to_num = dict((c, i) for i, c in enumerate(chars))

# Checking
inputs_len = len(processed_inputs)
vocab_len = len(chars)
print("Totel number of characters:", inputs_len)
print("Total vocab", vocab_len)

seq_length = 100
x_data = []
y_data = []

# Loop through the sequence
for i in range(0, inputs_len - seq_length, 1):
    in_seq = processed_inputs(i:i + seq_length)
    out_seq = processed_inputs(i + seq_length)
    x_data.append([chars_to_num[char] for char in in_seq])
    y.data.appened(chars_to_num[out_seq])

n_patters = len(x_data)
print("Total Patters:", n_patters)


# Converting the statements to numpy array
x = numpy.reshape(x_data,(n_patters, seq_length, 1))
x = x/float(vocab_len)


# One - hot Encoding
y = np_utils.to_categorical(y_data)
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0,2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0,2))
model.add(LSTM(128))
model.add(Dropout(0,2))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Saving the weights

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]
model.fit(x, y, epochs=4, batch_size=256, callbacks=desired_callbacks)


# Recompile the model with the saved weights
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adams')


# Output of the model back into characters
num_to_char = dict((i,c) for i,c in enumerate(chars))

# Random seed to generate
start = numpy.random.randint(0, len(x_data)-1)
pattern = x_data[start]
print("Random seed: ")
print("\".".join([num_to_char[value] for value in pattern]))

# Generate Text
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x/float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
