import cv2
import numpy as np
import math

from random import*

class Augment(object):              # hcw v1
#======================================================#
    #--------------------------------------------------#
    def cal_augment(self, img):
        
        #number = randint(1,1)
        img = np.array(img)
        
        h, w, _ = img.shape
        
        case_d = randint(0,3)
        
        #case_c = randint(0,1)
        
        #case_n = randint(0,100)
        
        # 191110
        #img = self.cropImage(img, 224, 224, -1, -1)
        
        # flip 
        if case_d == 0:
            img = cv2.flip(img, 0)
            #print('flip 1 ')
        # flip
        elif case_d == 1:
            img = cv2.flip(img, 1)
            #print('flip 2 ')
        elif case_d == 2:
            img = img
            #print('flip x')
        elif case_d == 3:
            img = self.set_rand_crop(img, 0.4)
            #cv2.imwrite('test.png', img)
            #print('random_crop')
        
#        # rotate
#        if case_c == 0:
#            img = self.set_rotate(img, 10) #default
#            #print('rotation')

##         elif case_n == 4:
##             img = self.set_color(img, 0.05)
##             #img = img
##             print('color')
#        elif case_c == 1:
#            img = self.set_rand_crop(img, 0.1)
#            #cv2.imwrite('test.png', img)
#            #print('random_crop')


                # rotate
        #if case_c == 1:
        #    img = self.set_rand_crop(img, 0.1)
        #    #cv2.imwrite('test.png', img)
        #    #print('random_crop')


        #if case_n < 5:
        #    img = self.set_gaussian(img)
        #    #print('gaussian')
                
        return img
    #--------------------------------------------------#
    def set_rotate(self, img, angle_range):

        h, w, _ = img.shape

        angle = randint(-angle_range, angle_range)

        M = cv2.getRotationMatrix2D((w/2, h/2), angle,1)
        img = cv2.warpAffine(img, M, (w, h))

        angle = math.fabs(angle)

        crop_h = int(h / (math.cos(math.pi * angle / 180) + math.sin(math.pi * angle / 180)))
        crop_w = int(w / (math.cos(math.pi * angle / 180) + math.sin(math.pi * angle / 180)))

        img = self.cropImage(img, crop_h, crop_w, -1, -1)
        #print('rotate ')
        #cv2.imwrite('test.png', img)
        return img
    #--------------------------------------------------#
    def cropImage(self,input, cal_crop_size_x, cal_crop_size_y, 
                  crop_center_x, crop_center_y):
        
        if len(np.shape(input)) == 3:
            h, w, _ = np.shape(input)
        else:
            h, w = np.shape(input)
    
        if crop_center_x == -1:
            h_center = int(h / 2)
            w_center = int(w / 2)
    
            h_delta = int(cal_crop_size_y / 2)
            w_delta = int(cal_crop_size_x / 2)
    
            sub_input = input[h_center - h_delta:h_center+ h_delta, 
                               w_center - w_delta:w_center+w_delta]
    
        else:
            h_center = crop_center_y
            w_center = crop_center_x
    
            h_delta = int(cal_crop_size_y / 2)
            w_delta = int(cal_crop_size_x / 2)
    
            sub_input = input[h_center - h_delta:h_center+ h_delta, 
                               w_center - w_delta:w_center+w_delta]


            #print(h_center - h_delta)
            #print(w_center - w_delta)
            #print(h_center + h_delta)
            #print(w_center + w_delta)
            #print('rand crop')
    
        return sub_input
    #--------------------------------------------------#
    def set_gaussian(self, img):

        #h, w, _ = img.shape
        sigma = uniform(0.5, 3.0)
        kernel = randint(3, 9)
        
        if kernel % 2 == 0:
            kernel = kernel + 1

        img = cv2.GaussianBlur(img, (kernel, kernel), sigma)

        #cv2.imwrite('test.png', img)
        return img
    #--------------------------------------------------#
    def set_resize(self, img, rate):

        h, w, _ = img.shape

        h_rate = int(h*rate)
        w_rate = int(w*rate)

        rand_h = randint(-h_rate, h_rate)
        rand_w = randint(-w_rate, w_rate)

        row_new = int(w + rand_w)
        col_new = int(h + rand_h)
        img = cv2.resize(img, (row_new, col_new), interpolation = cv2.INTER_CUBIC)

        if row_new > col_new:
            crop_new = col_new
        else:
            crop_new = row_new

        img = self.cropImage(img, crop_new, crop_new, -1, -1)

        #cv2.imwrite('test.png', img)
        return img
    #--------------------------------------------------#
    def set_color(self, img, rate):

        h, w, _ = img.shape

        b = img[:,:,0]
        g = img[:,:,1]
        r = img[:,:,2]

        b_min = np.min(b)
        b_max = np.max(b)
        b_range = int((b_max - b_min) * rate)
        b_rand = randint(-b_range, b_range)
        b = b + b_rand

        g_min = np.min(g)
        g_max = np.max(g)
        g_range = int((g_max - g_min) * rate)
        g_rand = randint(-g_range, g_range)
        g = g + g_rand

        r_min = np.min(r)
        r_max = np.max(r)
        r_range = int((r_max - r_min) * rate)
        r_rand = randint(-r_range, r_range)
        r = r + r_rand

        buffer = np.zeros((h, w, 3), dtype = np.int16)

        buffer[:,:,0] = b
        buffer[:,:,1] = g
        buffer[:,:,2] = r

        buffer[buffer < 0] = 0
        buffer[buffer > 255] = 255

       # cv2.imwrite('test.png', buffer)
        return buffer
    #--------------------------------------------------#
    def set_rand_crop(self, img, rate):

            
        h, w, _ = img.shape                             # ??? h, w = 512 ???? => 253 으로 수정
        #print(h, w)

        center_h = int(h / 2)
        center_w = int(w / 2)

        h_rate = int(rate * h )
        w_rate = int(rate * w )


        delta_h = int(rate * h / 2)
        delta_w = int(rate * w / 2)

        delta_h = randint(-delta_h, delta_h)
        delta_w = randint(-delta_w, delta_w)

        new_size_h = h - (abs(h_rate)) - 4
        new_size_w = w - (abs(w_rate)) - 4

        #print(new_size_h)
        #print(new_size_w)

        new_center_h = center_h + int(delta_h)
        new_center_w = center_w + int(delta_w)

        img = self.cropImage(img, new_size_w, new_size_h, new_center_w, new_center_h)
        #print(img.shape)
        
        
        #img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
        
        return img

    def set_contrast(self, img, alpha_r, beta_r):
        
        alpha = uniform(0.8, 1.2)
        beta = randint(0, 80)

        img = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)
        #new_img = np.zeros(img.shape)

        #for y in range(img.shape[0]):
        #    for x in range(img.shape[2]):
        #        for c in range(img.shape[2]):
        #            new_img[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
        return img
#======================================================#





#class Augment(object):             ## original
#======================================================#
#    --------------------------------------------------#
#    def cal_augment(self, img):
        
#        number = randint(0,5)
#        img = np.array(img)

#        for i in range(number):

#            h, w, _ = img.shape
#            case_n = randint(0, 5)
#             flip 
#            if case_n == 0:
#                img = cv2.flip(img, 0)
#                print('flip 1 ')
#             flip
#            elif case_n == 1:
#                img = cv2.flip(img, 1)
#                print('flip 2 ')
#             rotate
#            elif case_n == 2:
#                img = self.set_rotate(img, 40) #default
#                img = self.set_rotate(img, 10)
#             blur
#            elif case_n == 3:
#                img = self.set_gaussian(img)
#                cv2.imwrite('D:\Classification1\Dataset\BOE17\A2_DDM_GRAY_SCAN_IMAGE\blur.png', img)

#            elif case_n == 4:
#                img = self.set_color(img, 0.05)
#                print('color')
#            elif case_n == 5:
#                img = self.set_rand_crop(img, 0.3)
#                cv2.imwrite('test.png', img)
                
#            elif case_n == 3:
#                img = img

#            elif case_n == 7:
#                img = self.set_contrast(img, 0.1, 10)
#                cv2.imwrite('D:\Classification1\Dataset\BOE17\A2_DDM_GRAY_SCAN_IMAGE\contrast.png', img)
#                cv2.imwrite('test.png', img)
#             resize
#            elif case_n == 7:
#                img = self.set_resize(img, 0.1)
#                print('resize')
#             color & intensity
#        cv2.imwrite('test.png', img)
#        return img
#    --------------------------------------------------#
#    def set_rotate(self, img, angle_range):

#        h, w, _ = img.shape

#        angle = randint(-angle_range, angle_range)

#        M = cv2.getRotationMatrix2D((w/2, h/2), angle,1)
#        img = cv2.warpAffine(img, M, (w, h))

#        angle = math.fabs(angle)

#        crop_h = int(h / (math.cos(math.pi * angle / 180) + math.sin(math.pi * angle / 180)))
#        crop_w = int(w / (math.cos(math.pi * angle / 180) + math.sin(math.pi * angle / 180)))

#        img = self.cropImage(img, crop_h, crop_w, -1, -1)
#        print('rotate ')
#        cv2.imwrite('test.png', img)
#        return img
#    --------------------------------------------------#
#    def cropImage(self,input, cal_crop_size_x, cal_crop_size_y, 
#                  crop_center_x, crop_center_y):
        
#        if len(np.shape(input)) == 3:
#            h, w, _ = np.shape(input)
#        else:
#            h, w = np.shape(input)
    
#        if crop_center_x == -1:
#            h_center = int(h / 2)
#            w_center = int(w / 2)
    
#            h_delta = int(cal_crop_size_y / 2)
#            w_delta = int(cal_crop_size_x / 2)
    
#            sub_input = input[h_center - h_delta:h_center+ h_delta, 
#                               w_center - w_delta:w_center+w_delta]
    
#        else:
#            h_center = crop_center_y
#            w_center = crop_center_x
    
#            h_delta = int(cal_crop_size_y / 2)
#            w_delta = int(cal_crop_size_x / 2)
    
#            sub_input = input[h_center - h_delta:h_center+ h_delta, 
#                               w_center - w_delta:w_center+w_delta]


#            print(h_center - h_delta)
#            print(w_center - w_delta)
#            print(h_center + h_delta)
#            print(w_center + w_delta)
#            print('rand crop')
    
#        return sub_input
#    --------------------------------------------------#
#    def set_gaussian(self, img):

#        h, w, _ = img.shape
#        sigma = uniform(0.5, 3.0)
#        kernel = randint(3, 9)
        
#        if kernel % 2 == 0:
#            kernel = kernel + 1

#        img = cv2.GaussianBlur(img, (kernel, kernel), sigma)

#        cv2.imwrite('test.png', img)
#        return img
#    --------------------------------------------------#
#    def set_resize(self, img, rate):

#        h, w, _ = img.shape

#        h_rate = int(h*rate)
#        w_rate = int(w*rate)

#        rand_h = randint(-h_rate, h_rate)
#        rand_w = randint(-w_rate, w_rate)

#        row_new = int(w + rand_w)
#        col_new = int(h + rand_h)
#        img = cv2.resize(img, (row_new, col_new), interpolation = cv2.INTER_CUBIC)

#        if row_new > col_new:
#            crop_new = col_new
#        else:
#            crop_new = row_new

#        img = self.cropImage(img, crop_new, crop_new, -1, -1)

#        cv2.imwrite('test.png', img)
#        return img
#    --------------------------------------------------#
#    def set_color(self, img, rate):

#        h, w, _ = img.shape

#        b = img[:,:,0]
#        g = img[:,:,1]
#        r = img[:,:,2]

#        b_min = np.min(b)
#        b_max = np.max(b)
#        b_range = int((b_max - b_min) * rate)
#        b_rand = randint(-b_range, b_range)
#        b = b + b_rand

#        g_min = np.min(g)
#        g_max = np.max(g)
#        g_range = int((g_max - g_min) * rate)
#        g_rand = randint(-g_range, g_range)
#        g = g + g_rand

#        r_min = np.min(r)
#        r_max = np.max(r)
#        r_range = int((r_max - r_min) * rate)
#        r_rand = randint(-r_range, r_range)
#        r = r + r_rand

#        buffer = np.zeros((h, w, 3), dtype = np.int16)

#        buffer[:,:,0] = b
#        buffer[:,:,1] = g
#        buffer[:,:,2] = r

#        buffer[buffer < 0] = 0
#        buffer[buffer > 255] = 255

#        cv2.imwrite('test.png', buffer)
#        return buffer
#    --------------------------------------------------#
#    def set_rand_crop(self, img, rate):

#        img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_CUBIC)
#        h, w, _ = img.shape                             # ??? h, w = 512 ????
#        print(h, w)

#        center_h = int(h / 2)
#        center_w = int(w / 2)

#        h_rate = int(rate * h )
#        w_rate = int(rate * w )


#        delta_h = int(rate * h / 2)
#        delta_w = int(rate * w / 2)

#        delta_h = randint(-delta_h, delta_h)
#        delta_w = randint(-delta_w, delta_w)

#        new_size_h = h - (abs(h_rate)) - 4
#        new_size_w = w - (abs(w_rate)) - 4

#        print(new_size_h)
#        print(new_size_w)

#        new_center_h = center_h + int(delta_h)
#        new_center_w = center_w + int(delta_w)

#        img = self.cropImage(img, new_size_w, new_size_h, new_center_w, new_center_h)
#        print(img.shape)
#        return img

#    def set_contrast(self, img, alpha_r, beta_r):
        
#        alpha = uniform(0.8, 1.2)
#        beta = randint(0, 80)

#        img = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)
#        new_img = np.zeros(img.shape)

#        for y in range(img.shape[0]):
#            for x in range(img.shape[2]):
#                for c in range(img.shape[2]):
#                    new_img[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
#        return img
#======================================================#

