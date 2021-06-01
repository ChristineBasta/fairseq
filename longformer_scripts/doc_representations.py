from  huggingface_models import LongformerUsage
import data_prepare_wmt
import torch
import os
import logging
import argparse

longFormer=LongformerUsage()
def get_doc_representation(folder, num):
    doc_data, x, y=data_prepare_wmt.read_doc_by_num(folder, num)
    print(doc_data)
    output = longFormer.get_output(doc_data)
    print(output)
    return output

def get_doc_representation(folder, extension):
    #,max no.of documents
    max_file_no = len(os.listdir(folder))
    file_numbers = []
    for filename in os.listdir(folder):
        file_num = filename.split('_')[0]
        file_numbers.append(int(file_num))
    docs_representations_mean_hidden_classify={}
    docs_representations_pooled_output_classify= {}
    docs_representations_mean_hidden_modeling= {}
    docs_representations_pooled_output_modeling = {}
    for num in file_numbers:
        doc_data, indices=data_prepare_wmt.read_doc_by_num(folder, num, extension)

        #as classifier
        outputs1, mean_sequence_output1, pooled_output1 = longFormer.get_output(doc_data, classify_model=True)
        #as model
        outputs2, mean_sequence_output2, pooled_output2 = longFormer.get_output(doc_data, classify_model=False)
        docs_representations_mean_hidden_classify[num] = mean_sequence_output1
        docs_representations_pooled_output_classify[num] = pooled_output1

        docs_representations_mean_hidden_modeling[num] = mean_sequence_output2
        docs_representations_pooled_output_modeling[num] = pooled_output2

    return docs_representations_mean_hidden_classify, docs_representations_pooled_output_classify, docs_representations_mean_hidden_modeling, docs_representations_pooled_output_modeling

def get_doc_representation_test_dev(docs_dic):
    #,max no.of documents
    #todo_check
    no_doc=len(docs_dic)
    doc_num=0
    docs_representations_mean_hidden_classify={}
    docs_representations_pooled_output_classify= {}
    docs_representations_mean_hidden_modeling= {}
    docs_representations_pooled_output_modeling = {}
    for num in range(1, no_doc+1):
        doc_num = num
        doc_data=docs_dic[doc_num]
        print(doc_data)
        #as classifier
        print(doc_data)
        if(len(doc_data.split())<3000):
            outputs1, mean_sequence_output1, pooled_output1 = longFormer.get_output(doc_data, classify_model=True)
            #as model
            outputs2, mean_sequence_output2, pooled_output2 = longFormer.get_output(doc_data, classify_model=False)
            docs_representations_mean_hidden_classify[doc_num] = mean_sequence_output1
            docs_representations_pooled_output_classify[doc_num] = pooled_output1

            docs_representations_mean_hidden_modeling[doc_num] = mean_sequence_output2
            docs_representations_pooled_output_modeling[doc_num] = pooled_output2

    return docs_representations_mean_hidden_classify, docs_representations_pooled_output_classify, docs_representations_mean_hidden_modeling, docs_representations_pooled_output_modeling

def save_tensors_docs(doc_tensors, file_name):
    torch.save(doc_tensors, file_name)

def retrieve_tensors_doc(file_name):
    loaded = torch.load(file_name)
    print(loaded)
    print(loaded[1])
    print(loaded[2])


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", help="extension")
    parser.add_argument("--folder_represent", help="the folder of training which we are going to have lf represenattions for each document")
    parser.add_argument("--save_reps", help="file to save representations")
    parser.add_argument("--kind_reps", help="integer for which kind of representations")
    parser.add_argument("--doc_dic_valid", help="file that contains dic for documents of valid")
    parser.add_argument("--doc_dic_test", help="file that contains dic for documents of test")
    parser.add_argument("--save_reps_valid", help="file to save representations of valid")
    parser.add_argument("--save_reps_test", help="file to save representations of test")
    parser.add_argument("--save_reps_all", help="file to save representations of all together")
    args = parser.parse_args()
    logger.info(args)
    extension = args.extension
    folder_represent = args.folder_represent
    save_reps = args.save_reps
    kind_reps = int(args.kind_reps)
    doc_text_valid_file=args.doc_dic_valid
    doc_text_test_file=args.doc_dic_test
    save_reps_valid=args.save_reps_valid
    save_reps_test=args.save_reps_test
    save_reps_all=args.save_reps_all


    #extension='en'
    #folder_represent='/home/christine/news_micro_v/English'
    #file_to_save='mean_hidden.h5'

    #training data part
    '''
    rep_mean_hidden_classify, rep_pooled_output_classify, reps_mean_hidden_modeling, reps_pooled_output_modeling=get_doc_representation(folder_represent, extension)
    if(kind_reps==1):
        torch.save(rep_mean_hidden_classify, save_reps)
    elif (kind_reps == 2):
        torch.save(rep_pooled_output_classify, save_reps)
    elif (kind_reps == 3):
        torch.save(reps_mean_hidden_modeling, save_reps)
    elif (kind_reps == 4):
        torch.save(reps_pooled_output_modeling, save_reps)

    #valid
    '''
    doc_text_valid_dict=torch.load(doc_text_valid_file)
    mean_classify_v, pooled_classify_v, mean_lm_v, pooled_lm_v=get_doc_representation_test_dev(doc_text_valid_dict)
    if (kind_reps == 1):
        torch.save(mean_classify_v, save_reps_valid)
    elif (kind_reps == 2):
        torch.save(pooled_classify_v, save_reps_valid)
    elif (kind_reps == 3):
        torch.save(mean_lm_v, save_reps_valid)
    elif (kind_reps == 4):
        torch.save(pooled_lm_v, save_reps_valid)

    #test
    '''
    doc_text_test_dict=torch.load(doc_text_test_file)
    mean_classify_t, pooled_classify_t, mean_lm_t, pooled_lm_t=get_doc_representation_test_dev(doc_text_test_dict)
    if (kind_reps == 1):
        torch.save(mean_classify_t, save_reps_test)
    elif (kind_reps == 2):
        torch.save(pooled_classify_t, save_reps_test)
    elif (kind_reps == 3):
        torch.save(mean_lm_t, save_reps_test)
    elif (kind_reps == 4):
        torch.save(pooled_lm_t, save_reps_test)

    #make a dictionary for train, test, valid
    dict_training=torch.load(save_reps)
    dict_valid= torch.load(save_reps_valid)
    dict_test = torch.load(save_reps_test)
    dict_all={}
    dict_all['train']=dict_training
    dict_all['valid']=dict_valid
    dict_all['test'] = dict_test

    torch.save(dict_all, save_reps_all)
    '''
