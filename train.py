'''
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, UpSampling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
'''
# A) Import thêm tiện ích
import os, time, json
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau





from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                      Dense, Reshape, Conv2DTranspose, UpSampling2D, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


from libs.read_images_from import read_images_from
import argparse
# import numpy as np
# import cv2


EPOCHS = 5000

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="",
    help="model name", required=True)
ap.add_argument("-n", "--name", type=str, default="",
    help="Image folder name", required=True)
args = vars(ap.parse_args())

image_size = 64
input_shape = (image_size, image_size, 3)
layer_filters = [32, 64]
kernel_size = 5
latent_dim = 16
batch_size = 128

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs

for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

shape = K.int_shape(x)

x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)
    x = UpSampling2D((2, 2))(x)


x = Conv2DTranspose(filters=3,
                    kernel_size=kernel_size,
                    padding='same')(x)
outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()




#B) Trước khi compile (hoặc ngay sau), tạo thư mục & logger path
os.makedirs("model", exist_ok=True)
os.makedirs("logs", exist_ok=True)
run_id = time.strftime("%Y%m%d-%H%M%S")
tb_logdir = os.path.join("logs", f"autoencoder_{run_id}")
csv_log = os.path.join("logs", f"history_{run_id}.csv")
ckpt_path = os.path.join("model", f"{args['model']}.best.weights.keras")




# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()
#optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
optimizer = Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.999)
autoencoder.compile(loss='mean_absolute_error', optimizer=optimizer)

wraped_face, a_faces = read_images_from("images/{0}".format(args["name"]))

a_faces = a_faces.astype('float32') / 255.
wraped_face = wraped_face.astype('float32') / 255.

# print(a_faces[0].shape)
# print(wraped_face[0].shape)
# cv2.imshow("wrap image", wraped_face[0])
# cv2.imshow("face image", a_faces[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# b_faces = read_images_from("images/rathanak")
# b_faces = b_faces.astype('float32') / 255.


history = autoencoder.fit(
    wraped_face,
    a_faces,
    epochs=EPOCHS,
    batch_size=batch_size,
    validation_split=0.2,                # <- thêm validation
    callbacks=[
        TensorBoard(log_dir=tb_logdir),
        CSVLogger(csv_log),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=False)
    ],
    shuffle=True
)




autoencoder.fit(wraped_face,
                a_faces,
                epochs=EPOCHS,
                batch_size=batch_size,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
'''
autoencoder.save("model/{0}_model.h5".format(args["model"]))
autoencoder.save_weights("model/{0}_weight.h5".format(args["model"]))
'''


# Lưu full model (SavedModel/.keras)
autoencoder.save(os.path.join("model", f"{args['model']}.keras"))

# Lưu cân nặng: PHẢI có đuôi .weights.h5
autoencoder.save_weights(os.path.join("model", f"{args['model']}.weights.h5"))

# (tùy chọn) lưu lịch sử train ra JSON để vẽ biểu đồ sau này
with open(os.path.join("logs", f"history_{run_id}.json"), "w", encoding="utf-8") as f:
    json.dump({k: list(map(float, v)) for k, v in history.history.items()}, f, ensure_ascii=False, indent=2)
