import tensorflow as tf

fname     = "data/HP1.txt"
textlines = open(fname,"r").readlines()
text      = " ".join(textlines)
vocab     = sorted(set(text))
print(f'{len(vocab)} unique characters')
print (vocab)

chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
chars = chars_from_ids(ids)

# Function
def text_from_ids(ids):
	return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

#### I think I need to build the predition sequences
all_ids     = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

print (ids_dataset)

for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

# This is where the data is actually held and processed
sequences = ids_dataset.batch(6, drop_remainder=True)

def split_input_target(sequence):
	print (sequence)
	# First block
	input_text = sequence[:-1]
	# Single character (in list)
	target_text = [sequence[-1]]
	print (input_text, target_text)
	return input_text, target_text


dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
	print("Input :", text_from_ids(input_example).numpy())
	print("Target:", text_from_ids(target_example).numpy())

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
    )

# Build model
# Here I want to take my sequence input and the adjacency matrix and multiply through so some structure is learned
# Then pass to dense layers
# And have softmax output
