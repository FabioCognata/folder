from tflite_model_maker import audio_classifier
import os
'''
data_dir = 'C:/Users/cogna/Desktop/final_dataset/'

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=2 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

train_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(data_dir, 'train'), cache=False)
train_data, validation_data = train_data.split(0.9)
test_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(data_dir, 'test'), cache=False)

print(train_data)

batch_size = 47
epochs = 100

print('Training the model')
model = audio_classifier.create(
    train_data,
    spec,
    validation_data,
    batch_size=batch_size,
    epochs=epochs)

#print('Evaluating the model')
#model.evaluate(test_data)

models_path = './'
print(f'Exporing the TFLite model to {models_path}')

model.export(models_path, tflite_filename='newest_car_model.tflite')
'''