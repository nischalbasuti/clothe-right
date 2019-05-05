#!/bin/env python3
import scipy.io as sio

def get_label_dict():
    label_list = sio.loadmat("./clothing-co-parsing/label_list.mat")['label_list']

    label_dict = {}
    count = 1
    for i in label_list[0]:
        # Subtract by 1 to compensate for matlab indices starting at 1.
        label_dict[ i[0] ] =  count - 1 
        count += 1
    print(label_dict)


label_dict = {'null': 0, 'accessories': 1, 'bag': 2, 'belt': 3, 'blazer': 4, 'blouse': 5, 'bodysuit': 6, 'boots': 7, 'bra': 8, 'bracelet': 9, 'cape': 10, 'cardigan': 11, 'clogs': 12, 'coat': 13, 'dress': 14, 'earrings': 15, 'flats': 16, 'glasses': 17, 'gloves': 18, 'hair': 19, 'hat': 20, 'heels': 21, 'hoodie': 22, 'intimate': 23, 'jacket': 24, 'jeans': 25, 'jumper': 26, 'leggings': 27, 'loafers': 28, 'necklace': 29, 'panties': 30, 'pants': 31, 'pumps': 32, 'purse': 33, 'ring': 34, 'romper': 35, 'sandals': 36, 'scarf': 37, 'shirt': 38, 'shoes': 39, 'shorts': 40, 'skin': 41, 'skirt': 42, 'sneakers': 43, 'socks': 44, 'stockings': 45, 'suit': 46, 'sunglasses': 47, 'sweater': 48, 'sweatshirt': 49, 'swimwear': 50, 't-shirt': 51, 'tie': 52, 'tights': 53, 'top': 54, 'vest': 55, 'wallet': 56, 'watch': 57, 'wedges': 58}

tops_dict = {
 'blazer': 4,
 'blouse': 5,
 'bodysuit': 6,
 'bra': 8,
 'cardigan': 11,
 'coat': 13, ##################
 'dress': 14, #################
 'hoodie': 22,
 # 'intimate': 23,
 'jacket': 24,
 'jumper': 26,
 'romper': 35,
 'scarf': 37,
 'shirt': 38,
 # 'skin': 41,
 'suit': 46,
 'sweater': 48,
 'sweatshirt': 49,
 # 'swimwear': 50,
 't-shirt': 51,
 'tie': 52,
 'top': 54,
 'vest': 55,
}

if __name__ == '__main__':
    get_label_dict()
