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

def get_doc_representation(folder):
    #,max no.of documents
    max_file_no = len(os.listdir(folder))
    docs_representations_mean_hidden_classify={}
    docs_representations_pooled_output_classify= {}
    docs_representations_mean_hidden_modeling= {}
    docs_representations_pooled_output_modeling = {}
    for num in range(1, max_file_no+1):
        doc_data, x, y=data_prepare_wmt.read_doc_by_num(folder, num)
        print(doc_data)
        #as classifier
        outputs1, mean_sequence_output1, pooled_output1 = longFormer.get_output(doc_data, classify_model=True)
        #as model
        outputs2, mean_sequence_output2, pooled_output2 = longFormer.get_output(doc_data, classify_model=False)
        docs_representations_mean_hidden_classify[num] = mean_sequence_output1
        docs_representations_pooled_output_classify[num] = pooled_output1

        docs_representations_mean_hidden_modeling[num] = mean_sequence_output2
        docs_representations_pooled_output_modeling[num] = mean_sequence_output2

    return docs_representations_mean_hidden_classify, docs_representations_pooled_output_classify, docs_representations_mean_hidden_modeling, docs_representations_pooled_output_modeling

def get_doc_representation_test_dev(docs_dic):
    #,max no.of documents
    #todo_check
    no_doc=len(docs_dic)
    docs_representations_mean_hidden_classify={}
    docs_representations_pooled_output_classify= {}
    docs_representations_mean_hidden_modeling= {}
    docs_representations_pooled_output_modeling = {}
    for num in range(1, docs_dic+1):
        doc_data=docs_dic[no_doc]
        print(doc_data)
        #as classifier
        outputs1, mean_sequence_output1, pooled_output1 = longFormer.get_output(doc_data, classify_model=True)
        #as model
        outputs2, mean_sequence_output2, pooled_output2 = longFormer.get_output(doc_data, classify_model=False)
        docs_representations_mean_hidden_classify[num] = mean_sequence_output1
        docs_representations_pooled_output_classify[num] = pooled_output1

        docs_representations_mean_hidden_modeling[num] = mean_sequence_output2
        docs_representations_pooled_output_modeling[num] = mean_sequence_output2

    return docs_representations_mean_hidden_classify, docs_representations_pooled_output_classify, docs_representations_mean_hidden_modeling, docs_representations_pooled_output_modeling

def save_tensors_docs(doc_tensors, file_name):
    torch.save(doc_tensors, file_name)

def retrieve_tensors_doc(file_name):
    loaded = torch.load(file_name)
    print(loaded)
    print(loaded[1])
    print(loaded[2])


if __name__ == "__main__":
    eng_directory = '/home/christine/news-commentary/aligned/German-English/English'
    get_doc_representation(eng_directory, 2)

    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng_dir", help="The english directory")
    parser.add_argument("--stats_file", help="statistics file to write the stats")

    args = parser.parse_args()
    logger.info(args)
    eng_directory = args.eng_dir
    stats_file = args.stats_file

    #save_directory = '/home/christine/news-commentary/aligned/German-English/trail_save'
    #save_tensors_docs(save_directory)
    #retrieve_tensors_doc(save_directory)

