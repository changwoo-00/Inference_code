import os
import numpy as np
import cv2
import copy
import tensorflow as tf

from random import*
from augment import*


class Utils(object):
    
    def __init__(self, config):

        self.cAugment = Augment()
        self.config = config



    def read_class_txt(self, path):                 # train.py 에서 클래스 갯수를 구하는 용도로만 사용되고 있음
        path = os.path.join(path, 'Map')
        print(path)
    
        path_class = os.path.join(path, 'ClassList.txt')
        
        # class txt 파일 읽기
        class_list = []
        with open(path_class, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                line_list = line.split()
                class_list.append(line_list[0])
    
        print(class_list)
        return class_list
    
    def shuffle_list(self, *ls):
        l = list(zip(*ls))
        shuffle(l)
        return list(zip(*l))

    def read_file_path_txt(self, path):
        path = os.path.join(path, 'Map')
        print(path)
    
        #path_class = os.path.join(path, 'ImageList.txt')
        path_class = os.path.join(path, 'MapTrain.txt')    
        files_list = []
        labels_list = []
        dict_data = {}
        with open(path_class, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                line_list = line.split()
                files_list.append(line_list[0])
                labels_list.append(line_list[1])
                
                #dict_data[line_list[0]] = line_list[1]
        #test = list(zip(files_list, labels_list))
        #test = np.random.shuffle(test)
        files_list, labels_list = self.shuffle_list(files_list, labels_list)
        print("file numsers :", len(files_list))
        return files_list, labels_list

    def read_file_path_txt2(self, path):
# path : -.txt 까지의 path
        #path = os.path.join(path, 'Map')
        #print(path)
    
        #path_class = os.path.join(path, 'ImageList.txt')
        #path_class = os.path.join(path, 'MapValidation.txt')    
        files_list = []
        labels_list = []
        dict_data = {}
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
     
                line_list = line.split()
                files_list.append(line_list[0])
                labels_list.append(line_list[1])
                
                #dict_data[line_list[0]] = line_list[1]
        #test = list(zip(files_list, labels_list))
        #test = np.random.shuffle(test)
        files_list, labels_list = self.shuffle_list(files_list, labels_list)
        print("file numsers :", len(files_list))
        return files_list, labels_list

    def read_file_path_txt3(self, path, k_fold, ith, class_num, oversampling):   
        
# hcw added for cross_validation (test), input : path, output : train, test, data, label
# path : Map 이 저장된 위치
# set 별 class 비율 고려함. 
# train, test(validation) set 구분후 train set에 대해서 oversampling (aug x)
        
        path_class = path
        
        files_list = []
        labels_list = []


        
        dict_data_list = {new_list: [] for new_list in range(class_num)}
        dict_label_list = {new_list: [] for new_list in range(class_num)}
        

        
        with open(path_class, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                line_list = line.split()
                
                dict_data_list[int(line_list[1])].append(line_list[0])
                dict_label_list[int(line_list[1])].append(line_list[1])


        ###### shuffle before test,train split    # 19/10/05 => xxxx cv에서 중복됨.
        #for i in range(class_num):
        #    shuffle(dict_data_list[i])

        


        total_num_of_ele = 0
        for i in range(class_num):
            total_num_of_ele += len(dict_label_list[i])


        d_unit = int(total_num_of_ele/k_fold)              # test(validation) set 크기


        test_data = []
        test_label = []
        train_data = []
        train_label = []

        dict_test_data = {new_list: [] for new_list in range(class_num)}
        dict_test_label = {new_list: [] for new_list in range(class_num)}
        dict_train_data = {new_list: [] for new_list in range(class_num)}
        dict_train_label = {new_list: [] for new_list in range(class_num)}


        for i in range(class_num):                      # class 종류에 대해서 loop, train set은 oversampling을 위해 dictionary로 저장. test set은 list로 저장

            d_unit = int(len(dict_label_list[i])/k_fold)

            if k_fold == ith + 1:                           # 마지막 test set 일 경우
                test_data.extend(dict_data_list[i][ith*d_unit:])
                test_label.extend(dict_label_list[i][ith*d_unit:])

                #train_data.extend(dict_data_list[i][0:ith*d_unit])
                #train_label.extend(dict_label_list[i][0:ith*d_unit])
                #dict_test_data[i] = dict_data_list[i][ith*d_unit:]
                #dict_test_label[i] = dict_label_list[i][ith*d_unit:]

                dict_train_data[i] = dict_data_list[i][0:ith*d_unit]
                dict_train_label[i] = dict_label_list[i][0:ith*d_unit]

            else:                                           
                test_data.extend(dict_data_list[i][ith*d_unit:(ith+1)*d_unit])
                test_label.extend(dict_label_list[i][ith*d_unit:(ith+1)*d_unit])

                #train_data.extend(dict_data_list[i][0:ith*d_unit])
                #train_data.extend(dict_data_list[i][(ith+1)*d_unit:])
                
                #train_label.extend(dict_label_list[i][0:ith*d_unit])
                #train_label.extend(dict_label_list[i][(ith+1)*d_unit:])

                #dict_test_data[i] = dict_data_list[i][ith*d_unit:(ith+1)*d_unit]
                #dict_test_label[i] = dict_label_list[i][ith*d_unit:(ith+1)*d_unit]

                dict_train_data[i] = dict_data_list[i][0:ith*d_unit]
                dict_train_data[i].extend(dict_data_list[i][(ith+1)*d_unit:])
                
                dict_train_label[i] = dict_label_list[i][0:ith*d_unit]
                dict_train_label[i].extend(dict_label_list[i][(ith+1)*d_unit:])





        ###### augmentation ###### test set fix, train set 만 over sampling

        class_size_list = [0]*class_num # will be added
        
        for i in range(class_num):
            
            class_size_list[i] = len(dict_train_label[i]) # will be added (augmentation)


        if oversampling :
            max_size_class = np.argmax(class_size_list) # will be added

            for i in range(class_num):

                class_i_len = len(dict_train_label[i])
                if i != max_size_class:
                    while len(dict_train_label[i]) != int(class_size_list[max_size_class]*oversampling) and len(dict_train_label[i]) < int(class_size_list[max_size_class]*oversampling):
                    #while len(dict_train_label[i]) != class_size_list[max_size_class]:
                        
                        j = randint(0, class_i_len-1)
                        #print('dict_data_list[i][j] : {}'.format(dict_data_list[i][j]))
                        dict_train_data[i].append(dict_train_data[i][j])
                        dict_train_label[i].extend(dict_train_label[i][j])
                        #print('dict_data_list[i] size : {}'.format(len(dict_data_list[i])))
                   
            #for i in range(class_num):
            #    print('class {} size : {}'.format(i, len(dict_train_label[i])))         
            
        ##########################
        for i in range(class_num):
            print('class {} size : {}'.format(i, len(dict_train_label[i])))         
            




        for i in range(class_num):                                      # train set dictionary to list
            train_data.extend(dict_train_data[i])
            train_label.extend(dict_train_label[i])


        print('Train data size : {}'.format(len(train_label)))
        print('Test data size : {}'.format(len(test_label)))



        ######## shuffle ########
        train_zip = list(zip(train_data, train_label))
        test_zip = list(zip(test_data, test_label))
        shuffle(train_zip)
        shuffle(test_zip)
        train_data, train_label = zip(*train_zip)
        test_data, test_label = zip(*test_zip)
        #########################

        
        
        
        ####### save Train, test data
        save_dir = self.config.rootdir + 'testfolder\\'+self.config.title
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        outf = open(save_dir + "\\" + "train_data_" + str(ith) + "th.txt" , "w")
        for i in range(len(train_label)):
            outf.write(str(train_data[i])+"\t"+train_label[i])
            outf.write("\n")
        outf.close()
        outf = open(save_dir + "\\" + "test_data_" + str(ith) + "th.txt" , "w")
        for i in range(len(test_label)):
            outf.write(str(test_data[i])+"\t"+test_label[i])
            outf.write("\n")
        outf.close()
        ####################################



        print()
        print()
        print(train_data)
        print()
        print()
        print(train_label)


        return train_data, train_label, test_data, test_label


    
    def make_labels(self, labels, num):
    
        labels_0or1 = []
    
        for i in range(len(labels)):
    
            label = np.zeros(num, dtype = np.int16)
            label[int(labels[i])] = 1
            labels_0or1.append(label)
        
        return labels_0or1                  # 2dim matrix (?)
        #print(labels_0or1)
    



    #def edgeDetection(self,img):
    #    rst = copy.deepcopy(img)
    #    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    #    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    #    rst = (sobelx * 0.5) + (sobely * 0.5)
    #    rst = cv2.convertscaleabs(rst)
    #    return rst

    #def load_image(self, paths):
    
    #    img_arr = []
    #    for path in paths:
    
    #        img = cv2.imread(path)
    #        img_arr.append(img)
        
    #    return img_arr ## 0806 backup


    def load_image(self, paths):
    
        img_arr = []
        counter = 1
        for path in paths:
    
            img = cv2.imread(path)
            rst = copy.deepcopy(img)

            rst = cv2.resize(rst,(299,299)) ## 0927 han

            #rst = cv2.Laplacian(rst,cv2.CV_64F)
            #sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
            #sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
            #rst = (sobelx * 0.5) + (sobely * 0.5)
            #cv2.imwrite('test.png', img)
            #cv2.imwrite('test%d.png'%counter, rst)
            #rst = cv2.convertscaleabs(rst)
            img_arr.append(rst)
            counter += 1
        #for i in range(len(img_arr)):
        #    cv2.imwrite('new_data\laplacian_img%d.png'%i, img[i])
        
        return img_arr ## 0806 edge test

    def set_batch_data(self, datas_in, datas_out, batch_num, h, w, b_augment, b_rand, data_num):
        
        batch_in = []
        batch_out = []

        arr_test = []
        for i in range(batch_num):

            if b_rand == True:
                n = randint(0, len(datas_in) - 1)

            else:
                n = data_num * batch_num + i

            data_in = datas_in[n]
            #data_out = datas_out[n]

            if b_augment == True:
                data_in = self.cAugment.cal_augment(data_in)            

            data_in = self.set_resize(data_in, h, w)
            
            #data_in = (data_in - np.mean(data_in))/np.std(data_in) # hcw added (test)
            
            batch_in.append(data_in)
            #batch_out.append(data_out)

        #print(arr_test)
        batch_in = np.array(batch_in)
        #batch_out = np.array(batch_out)
        batch_in = batch_in.reshape(batch_num, h, w, 3)
        #batch_out = batch_out.reshape(batch_num, len(batch_out[0]))

        return batch_in, batch_out

    def set_batch_data2(self, datas_in_path, datas_in, datas_out, batch_num, h, w, b_augment, b_rand, data_num):
        
        batch_in_path = []
        batch_in = []
        batch_out = []

        arr_test = []

        print(np.shape(datas_in))

        for i in range(batch_num):

            if b_rand == True:
                n = randint(0, len(datas_in) - 1)

            else:
                n = data_num * batch_num + i

            data_in_path = datas_in_path[n]
            data_in = datas_in[n]
            data_out = datas_out[n]

            if b_augment == True:
                data_in = self.cAugment.cal_augment(data_in)            

            data_in = self.set_resize(data_in, h, w)
            
            #data_in = (data_in - np.mean(data_in))/np.std(data_in) # hcw added (test)
            
            batch_in_path.append(data_in_path)
            batch_in.append(data_in)
            batch_out.append(data_out)

        #print(arr_test)
        batch_in = np.array(batch_in)
        batch_out = np.array(batch_out)
        batch_in = batch_in.reshape(batch_num, h, w, 3)
        batch_out = batch_out.reshape(batch_num, len(batch_out[0]))

        return batch_in_path, batch_in, batch_out

    def set_resize(self, img, h, w):
        try : # hcw
            img = cv2.resize(img, (w, h), interpolation = cv2.INTER_CUBIC)
        except Exception as e:
            print(str(e))

        return img

#def get_files_path(path, file_type):

#    for file in os.listdir(path):
#        if file.endswith(file_type):
#            print(file)


