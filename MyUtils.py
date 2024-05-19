"""
 Created By Hamid Alavi on 7/3/2019
"""
import numpy as np
import keras
from medpy.io import load as load_mhd_images
import os.path as path
import cv2
from keras import backend as K


def split_train_test_id(id_all_data, testing_share):
    _id = id_all_data.copy()
    r = np.random.RandomState(seed=1000)
    r.shuffle(_id)
    id_training_data = np.sort(_id[np.floor(len(id_all_data) * testing_share).astype(np.uint16):])
    id_testing_data = np.sort(_id[:np.floor(len(id_all_data) * testing_share).astype(np.uint16)])
    return id_training_data, id_testing_data


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, image_ids, batch_size=32, image_resize_to=(480, 640), image_viewport=('2CH', '4CH'),
                 image_phase=('ES', 'ED'), image_masks_index=(0, 1, 2, 3) , shuffle=True):
        self.data_path = data_path
        self.image_ids = image_ids.copy()
        self.batch_size = batch_size
        self.image_resize_to = image_resize_to
        self.image_masks_index = image_masks_index
        self.shuffle = shuffle
        self.image_viewport = image_viewport
        self.image_phase = image_phase
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __len__(self):
        return int(np.floor(len(self.image_ids)/self.batch_size))

    def _load_data(self, image_id, image_viewport, image_phase):
        patient_id = 'patient%s' % str(image_id).zfill(4)
        image_path = path.join(self.data_path, patient_id, patient_id + '_%s_%s.mhd' % (image_viewport, image_phase))
        mask_path = path.join(self.data_path, patient_id, patient_id + '_%s_%s_gt.mhd' % (image_viewport, image_phase))
        _image, _ = load_mhd_images(image_path)
        _image = np.squeeze(_image)
        _image = cv2.resize(_image, self.image_resize_to[::-1])
        _image = np.expand_dims(_image, axis=-1)
        _image = _image/255
        _mask, _ = load_mhd_images(mask_path)
        _mask = np.squeeze(_mask)
        _mask = self._layers_single2multi(_mask)
        return _image, _mask

    def _layers_single2multi(self, mask):
        new_mask = np.empty((*self.image_resize_to, len(self.image_masks_index)), dtype=np.bool_)
        for i, layer_index in enumerate(self.image_masks_index):
            _layer = mask == layer_index
            _layer = cv2.resize(_layer.astype(np.uint8), self.image_resize_to[::-1], interpolation=cv2.INTER_NEAREST)
            new_mask[:, :, i] = _layer
        return new_mask
    
    def _layers_multi2single(self, mask):
        new_mask_shape = list(mask.shape)
        new_mask_shape[-1] = 1
        new_mask = np.empty(new_mask_shape, dtype=np.uint8)
        for j in range(mask.shape[0]):
            for i in range(mask.shape[-1]):
                new_mask[j, mask[j, :, :, i] == 1, 0] = i
        return new_mask    
        
    def __getitem__(self, batch_index):
        if batch_index == self.__len__() - 1:
            _image_ids = self.image_ids[batch_index * self.batch_size:]
        else:
            _image_ids = self.image_ids[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        images = []
        masks = []
        for _id in _image_ids:
            for vp in self.image_viewport:
                for ph in self.image_phase:
                    _image, _mask = self._load_data(_id, image_viewport=vp, image_phase=ph)
                    images.append(_image)
                    masks.append(_mask)
        images = np.array(images)
        masks = np.array(masks)
        return images, masks


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_pred = K.greater_equal(y_pred,0.5)
    y_pred = K.cast(y_pred, dtype=K.floatx())   
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)