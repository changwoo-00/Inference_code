import argparse
import tensorflow as tf
import sys
from utils import*
import time

args_list = []
parser = argparse.ArgumentParser()

def add_arg_group(name):
    """
    :param name: argument group, str
    :return: list (argument)
    """
    arg = parser.add_argument_group(name)
    args_list.append(arg)
    return arg

def get_config():
    cfg, un_parsed = parser.parse_known_args()
    return cfg, un_parsed

def get_placeholder(model):
    if model == 'inception_resnet_v2':
        input = graph.get_operation_by_name('image_input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        is_training = graph.get_operation_by_name('is_training').outputs[0]
        #logits = graph.get_operation_by_name('InceptionResnetV2/Logits/Logits').outputs[0]
        predictions = graph.get_operation_by_name('InceptionResnetV2/Logits/Predictions').outputs[0]
    elif model == 'resnet_v2_50':
        input = graph.get_operation_by_name('Placeholder').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        is_training = graph.get_operation_by_name('is_training').outputs[0]
        predictions = graph.get_operation_by_name('resnet_v2_50/predictions/Softmax').outputs[0]
    elif model== 'mobilenet_v2':
        input = graph.get_operation_by_name('image_input').outputs[0]
        #input = graph.get_operation_by_name('file_input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        is_training = graph.get_operation_by_name('is_training').outputs[0]
        predictions = graph.get_operation_by_name('MobilenetV2/Predictions/Reshape_1').outputs[0]
    return input, keep_prob, is_training, predictions

mode_arg = add_arg_group('mode')
mode_arg.add_argument('--title', type=str, default="mobilenet_v2")
mode_arg.add_argument('--batch_size', type=int, default=8)
mode_arg.add_argument('--meta_file_path_infer', type=str, default="D:\\HDL\\Project\\Classification\\191216_MOBILENETV2_TEST_2\\CheckPoint\\R90_C0.7_300EP\\0199-0152\\CLASSIFICATION.meta")
mode_arg.add_argument('--img_dir_path_infer', type=str, default='D:\\0.8Samsung\\trainset\\Classification_data\\train_data\\For_inference\\crop(1-A-28-1_)\\')

####
height = 299
width = 299
####


if __name__ == "__main__":

    config, _ = get_config()

    cUtils = Utils(config)
    class_num = 8
    i = 0
    path = config.img_dir_path_infer
    test_file_paths = os.listdir(config.img_dir_path_infer)
    test_file_paths = [path + line for line in test_file_paths if line[-3:] == 'jpg' or line[-3:] == 'png' or line[-3:] == 'bmp']


    metafilepath = config.meta_file_path_infer   # dataset v13  

    sys.setrecursionlimit(2000)

    test_labels_num = None
    if test_labels_num == None:
        test_labels_num = [1]*len(test_file_paths)

        
    #test_labels_0or1 = cUtils.make_labels(test_labels_num, len(class_names))
    test_labels_0or1 = None
    test_data = cUtils.load_image(test_file_paths)
    print('test file path')
    print(test_file_paths)
        
    print('test_data shape')
    print(np.shape(test_data))

    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(metafilepath) 
    ckptpath = metafilepath[:-5]
    print(ckptpath)








    with tf.Session() as sess:

        imported_meta.restore(sess, metafilepath[:-5])


        graph = tf.get_default_graph()

        input, keep_prob, is_training, predictions = get_placeholder(config.title)
        ############## data.dataset test ####################
        
        #x = tf.placeholder(tf.float32, shape=[None,2])
        dataset = tf.data.Dataset.from_tensor_slices(input).repeat().batch(config.batch_size)
        iter = dataset.make_initializable_iterator() # create the iterator
        batch_i = iter.get_next()
        #batch_j = batch_i.eval()
        sess.run(iter.initializer, feed_dict={input: test_data[:8]}) 
        #sess.run(iter.initializer) 
        print(np.shape(batch_i))
        print(sess.run(batch_i)) # output [ 0.52374458  0.71968478]

        #predict_list = []
        predict_list_2 = []
        batch_time_list = []

        dict_1= {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[]}

        #sess.run(iter.initializer, feed_dict={input: test_data[1], is_training: False})
        
        print('hi')
        
        for j in range(int(np.ceil(len(test_data)/config.batch_size))):
            batch_j = batch_i.eval()    
            print(batch_i)
            batch_start_time = time.time()
            temp_batch_size = config.batch_size
            if len(test_data)%config.batch_size != 0 and j == int(np.ceil(len(test_data)/config.batch_size)-1):
                temp_batch_size = len(test_data)%config.batch_size

        
            #batch_in, batch_out = cUtils.set_batch_data(test_data, test_labels_0or1, temp_batch_size, height, width, False, False, j)
            ##batch_in, batch_out = test_data, test_labels_0or1
            #path_temp = test_file_paths[j*temp_batch_size:(j+1)*temp_batch_size]
                
            
            ##### inception resnet v2 #########
            #predictions_o = sess.run([predictions], {input : batch_in, keep_prob : 1, is_training : False}) # test set 모두 사용하면서 cal_accuracy 평균은 틀린값이 됨.
            #####################
            ##### mobilenet #####
            #predictions_o = sess.run([predictions], {input : batch_in, keep_prob : 1, is_training : False}) # test set 모두 사용하면서 cal_accuracy 평균은 틀린값이 됨.
            #predictions_o = sess.run([predictions], {input : batch_i, is_training : False}) # test set 모두 사용하면서 cal_accuracy 평균은 틀린값이 됨.
            

            predictions_o = sess.run([predictions], {input : batch_j, is_training : False})

            #####################
                
            print(predictions_o)

            ##############################################################
            #thres = 0.7
            #for i in range(len(predictions_o[0])):
            #    #arg_sort_list = predictions_o[i].argsort()
            #    #if 0 in arg_sort_list[-2:] or 1 in arg_sort_list[-2:] or 2 in arg_sort_list[-2:]:
            #    #    predict_list.append(path_temp[i])
            #    #print(predictions_o[i])
            #    if predictions_o[0][i][0] > thres or predictions_o[0][i][1] > thres or predictions_o[0][i][2] > thres:
            #        predict_list.append(path_temp[i])
            #        #if predictions_o[0][i][0] > 0.999 or predictions_o[0][i][1] > 0.999 or predictions_o[0][i][2] > 0.999:
            #        #    predict_list.append(path_temp[i])
            #        #else:
            #        #    predict_list_2.append(path_temp[i])
            #        #    print(path_temp[i])
            #        #    print(predictions_o[0][i])
            #################################################################




            batch_end_time = time.time()

            batch_time = batch_end_time - batch_start_time
            print('batch_tact : ')
            print(batch_time)
            if j != int(np.ceil(len(test_data)/config.batch_size)-1):
                batch_time_list.append(batch_time)


        print('predict_list')
        print(predict_list)
        print(len(predict_list))
            
        print('tact time mean')
        print(batch_time_list[1:])
        print(np.mean(batch_time_list[1:]))

        #print('start file copy')
        #for i in predict_list:
        #    shutil.copyfile(i, 'D:\\0.8Samsung\\trainset\\Classification_data\\train_data\\For_inference\\classified\\191216_MOBILENETV2_TEST_UNDER20EP\\top\\' + i.split('\\')[-1])
