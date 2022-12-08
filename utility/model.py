import keras
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, \
    Activation, Multiply, Dense, GlobalAveragePooling2D, AveragePooling2D, Conv2DTranspose, DepthwiseConv2D, add, \
    SeparableConv2D, ZeroPadding2D
# from keras.optimizers import Adam
from keras import optimizers
import keras.backend as K
from keras.losses import categorical_crossentropy
import keras.layers as KL
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
from keras.utils import conv_utils
from utility.switchnorm import SwitchNormalization
import tensorflow as tf


def generalized_dice(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 4))
    y_pred = K.reshape(y_pred, shape=(-1, 4))
    sum_p = K.sum(y_pred, -2)
    sum_r = K.sum(y_true, -2)
    sum_pr = K.sum(y_true * y_pred, -2)
    weights = K.pow(K.square(sum_r) + K.epsilon(), -1)
    generalized_dice = (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))
    return generalized_dice


def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice(y_true, y_pred)


def custom_loss(y_true, y_pred):
    return 1 * generalized_dice_loss(y_true, y_pred) + 1 * categorical_crossentropy(y_true, y_pred)

# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal',
                         use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])


# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])

#空洞空间金字塔池化
def ASPP(x, dim, out_shape):
    b0 = Conv2D(dim, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(dim, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(dim, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(18, 18), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(dim, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(dim, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = Conv2D(dim, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(out_shape, out_shape))(b4))

    x = concatenate([b4, b0, b1, b2, b3])
    y = Conv2D(filters=dim, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    y = BatchNormalization(axis=3, gamma_regularizer=regularizers.l2(1e-4), beta_regularizer=regularizers.l2(1e-4))(y)
    y = Activation('relu')(y)
    return y

#深度可分离注意力残差块
def DeepResBlock(input, nb_filter, with_conv_shortcut=False):
    conv1 = SeparableConv2D(nb_filter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)

    SN1 = SwitchNormalization(axis=-1)(conv1)

    conv2 = SeparableConv2D(nb_filter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(SN1)

    SN2 = SwitchNormalization(axis=-1)(conv2)

    conv3 = SeparableConv2D(nb_filter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(SN2)
    SN3 = SwitchNormalization(axis=-1)(conv3)
    cbam1 = cbam(input, 0.5)
    if with_conv_shortcut:
        shortcut = BatchNormalization()(
            SeparableConv2D(nb_filter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cbam1))
        x = add([SN3, shortcut])
        return x
    else:
        x = add([SN3, cbam1])
        return x


def unet(pretrained_weights, input_size=(128, 128, 3), classNum=7, learning_rate=1e-5):

    inputs = Input(input_size)

    #  2D卷积层
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))

    res1_1 = DeepResBlock(conv1, 64)
    res1_2 = DeepResBlock(res1_1, 64)
    res1_3 = DeepResBlock(res1_2, 64)

    #  对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(res1_3)

    res2_1 = DeepResBlock(pool1, 128, with_conv_shortcut=True)
    res2_2 = DeepResBlock(res2_1, 128)
    res2_3 = DeepResBlock(res2_2, 128)
    res2_4 = DeepResBlock(res2_3, 128)

    pool2 = MaxPooling2D(pool_size=(2, 2))(res2_4)

    res3_1 = DeepResBlock(pool2, 256, with_conv_shortcut=True)
    res3_2 = DeepResBlock(res3_1, 256)
    res3_3 = DeepResBlock(res3_2, 256)
    res3_4 = DeepResBlock(res3_3, 256)
    res3_5 = DeepResBlock(res3_4, 256)
    res3_6 = DeepResBlock(res3_5, 256)

    pool3 = MaxPooling2D(pool_size=(2, 2))(res3_6)

    res4_1 = DeepResBlock(pool3, 512, with_conv_shortcut=True)
    res4_2 = DeepResBlock(res4_1, 512)
    res4_3 = DeepResBlock(res4_2, 512)

    #  Dropout正规化，防止过拟合
    drop4 = Dropout(0.5)(res4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


    res5_1 = DeepResBlock(pool4, 1024, with_conv_shortcut=True)
    drop5_1 = Dropout(0.5)(res5_1)

    cbam5_1 = cbam(drop5_1, 0.5)

    res5_2 = DeepResBlock(cbam5_1, 1024)
    drop5_2 = Dropout(0.5)(res5_2)


    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5_2))

    skip4_1 = cbam(drop4, 0.5)
    try:
        merge6_1 = concatenate([drop4, skip4_1, up6], axis=3)
    except:
        print('111')
        # merge6_1 = merge([drop4, skip4_1, up6], mode='concat', concat_axis=3)

    res6_1 = DeepResBlock(merge6_1, 512, with_conv_shortcut=True)
    res6_2 = DeepResBlock(res6_1, 512, with_conv_shortcut=True)

    drop6_1 = Dropout(0.5)(res6_2)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop6_1))

    skip3_1 = cbam(res3_6, 0.5)

    try:
        merge7_1 = concatenate([res3_6, skip3_1, up7], axis=3)
    except:
        print('111')
        # merge7_1 = merge([res3_6, skip3_1, up7], mode='concat', concat_axis=3)

    res7_1 = DeepResBlock(merge7_1, 256, with_conv_shortcut=True)
    res7_2 = DeepResBlock(res7_1, 256, with_conv_shortcut=True)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(res7_2))

    skip2_1 = cbam(res2_4, 0.5)
    try:
        merge8_1 = concatenate([res2_4, skip2_1, up8], axis=3)
    except:
        print('111')
        # merge8_1 = merge([res2_4, skip2_1, up8], mode='concat', concat_axis=3)

    res8_1 = DeepResBlock(merge8_1, 128, with_conv_shortcut=True)
    res8_2 = DeepResBlock(res8_1, 128, with_conv_shortcut=True)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(res8_2))

    skip1_1 = cbam(res1_3, 0.5)
    try:
        merge9 = concatenate([res1_3, skip1_1, up9], axis=3)
    except:
        print('111')
        # merge9 = merge([res1_3, skip1_1, up9], mode='concat', concat_axis=3)
    res9_1 = DeepResBlock(merge9, 64, with_conv_shortcut=True)
    res9_2 = DeepResBlock(res9_1, 64, with_conv_shortcut=True)

    conv10 = Conv2D(classNum, 1, activation='softmax')(res9_2)

    model = Model(inputs=inputs, outputs=conv10)

    #model.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss, metrics=['accuracy'])
    model.compile(optimizer="adam", loss=custom_loss, metrics=['accuracy'])


    #  如果有预训练的权重
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model