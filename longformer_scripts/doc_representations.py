from huggingface_models import LongformerUsage
import data_prepare_wmt
import torch
import os
import logging
import argparse
import h5py

longFormer = LongformerUsage()


def get_doc_representation(folder, num):
    doc_data, x, y = data_prepare_wmt.read_doc_by_num(folder, num)
    print(doc_data)
    output = longFormer.get_output(doc_data)
    print(output)
    return output


def get_doc_representation(folder, extension, kind_reps):
    # ,max no.of documents
    file_numbers = []
    for filename in os.listdir(folder):
        file_num = filename.split('_')[0]
        file_numbers.append(int(file_num))
    docs_representations = {}

    for num in file_numbers:
        doc_data, indices = data_prepare_wmt.read_doc_by_num(folder, num, extension)
        if (kind_reps == 1 or kind_reps == 2):
            representation = get_representations_classify(doc_data, kind_reps)
        elif (kind_reps == 3 or kind_reps == 4):
            representation = get_representations_model(doc_data, kind_reps)
        elif kind_reps == 5:
            representation = get_representations_seq_classify(doc_data)
        elif kind_reps == 6:
            representation = get_representations_masked_LM(doc_data)
        #docs_representations[num] = representation
        del representation
        torch.cuda.empty_cache()
    return docs_representations


def get_doc_representation_test_dev(docs_dic, kind_reps):
    # ,max no.of documents
    # todo_check
    no_doc = len(docs_dic)
    doc_num = 0
    docs_representations = {}

    for num in range(1, no_doc + 1):
        doc_num = num
        doc_data = docs_dic[doc_num]
        print(doc_data)
        # as classifier
        print(doc_data)
        if (len(doc_data.split()) < 3000):
            if (kind_reps == 1 or kind_reps == 2):
                representation = get_representations_classify(doc_data, kind_reps)
            elif (kind_reps == 3 or kind_reps == 4):
                representation = get_representations_model(doc_data, kind_reps)
            elif kind_reps == 5:
                representation = get_representations_seq_classify(doc_data)
            elif kind_reps == 6:
                representation = get_representations_masked_LM(doc_data)
            docs_representations[doc_num] = representation
            del representation
            torch.cuda.empty_cache()

    return docs_representations


def get_representations_classify(doc_data, kind_reps):
    outputs1, mean_sequence_output1, pooled_output1 = longFormer.get_output(doc_data, classify_model=True)
    if kind_reps == 1:
        return mean_sequence_output1
    elif kind_reps == 2:
        return pooled_output1


def get_representations_model(doc_data, kind_reps):
    outputs1, mean_sequence_output1, pooled_output1 = longFormer.get_output(doc_data, classify_model=False)
    if kind_reps == 3:
        return mean_sequence_output1
    elif kind_reps == 4:
        return pooled_output1


def get_representations_seq_classify(doc_data):
    outputs, mean_sequence_output = longFormer.get_output_seq_classification(doc_data)
    return mean_sequence_output


def get_representations_masked_LM(doc_data):
    outputs, mean_sequence_output = longFormer.get_output_maskedLM(doc_data)
    return mean_sequence_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", help="extension")
    parser.add_argument("--folder_represent",
                        help="the folder of training which we are going to have lf represenattions for each document")
    #parser.add_argument("--save_reps", help="file to save representations")
    parser.add_argument("--kind_reps", help="integer for which kind of representations")
    parser.add_argument("--doc_dic_valid", help="file that contains dic for documents of valid")
    parser.add_argument("--doc_dic_test", help="file that contains dic for documents of test")
    #parser.add_argument("--save_reps_valid", help="file to save representations of valid")
    #parser.add_argument("--save_reps_test", help="file to save representations of test")
    #parser.add_argument("--save_reps_all", help="file to save representations of all together")
    parser.add_argument("--which_file_reps", help="integer, 1 for train, 2 for valid, 3 for test and 4 for all")
    args = parser.parse_args()
    logger.info(args)
    extension = args.extension
    folder_represent = args.folder_represent
    #save_reps = args.save_reps
    kind_reps = int(args.kind_reps)
    doc_text_valid_file = args.doc_dic_valid
    doc_text_test_file = args.doc_dic_test
    #save_reps_valid = args.save_reps_valid
    #save_reps_test = args.save_reps_test
    #save_reps_all = args.save_reps_all
    which_file_reps=int(args.which_file_reps)


    # training data part
    if(which_file_reps==1):
        representations_dic=get_doc_representation(folder_represent, extension, kind_reps)
        file_name_start='train'
    elif (which_file_reps == 2):
        doc_text_valid_dict = torch.load(doc_text_valid_file)
        representations_dic = get_doc_representation_test_dev(doc_text_valid_dict, kind_reps)
        file_name_start = 'valid'
    elif (which_file_reps == 3):
        doc_text_test_dict = torch.load(doc_text_test_file)
        representations_dic = get_doc_representation_test_dev(doc_text_test_dict, kind_reps)
        file_name_start = 'test'

    if (kind_reps == 1):
        file_name_end='_mean_classify.h5'
    elif (kind_reps == 2):
        file_name_end = '_pooled_classify.h5'
    elif (kind_reps == 3):
        file_name_end = '_mean_model.h5'
    elif (kind_reps == 4):
        file_name_end = '_pooled_model.h5'
    elif (kind_reps == 5):
        file_name_end = '_seq_classify.h5'
    elif (kind_reps == 6):
        file_name_end = '_lm_masked.h5'

    #open and write a file if exits, else create
    #processed = h5py.File(file_name_start+file_name_end, 'a')
    #processed.create_dataset(str(num), data=representation, dtype='float32')
    torch.save(file_name_start+file_name_end, representations_dic)

    # make a dictionary for train, test, valid
    '''
    # todo(christine) to do all things together
    dict_training=torch.load(save_reps)
    dict_valid= torch.load(save_reps_valid)
    dict_test = torch.load(save_reps_test)
    dict_all={}
    dict_all['train']=dict_training
    dict_all['valid']=dict_valid
    dict_all['test'] = dict_test

    torch.save(dict_all, save_reps_all)
    '''

