from time import time
from scipy.io import wavfile

from wavenet.utils import make_batch
from wavenet.models import Model, Generator


inputs, targets = make_batch('assets/voice.wav')
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 0.8

print('Building model')
model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction)


print('Starting training')
tic = time()
model.train(inputs, targets)
toc = time()

print('Training took {} seconds.'.format(toc-tic))

generator = Generator(model)

# Get first sample of input
input_ = inputs[:, 0:1, 0]

tic = time()
predictions = generator.run(input_)
toc = time()
print('Generating took {} seconds.'.format(toc-tic))
wavfile.write('predicted.wav', rate=44100, predictions=predictions)
