from  huggingface_trials import LongformerUsage
import data_prepare_wmt
import torch
import os

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
    docs_representations={}
    for num in range(1, max_file_no+1):
        doc_data, x, y=data_prepare_wmt.read_doc_by_num(folder, num)
        print(doc_data)
        last_hidden_layer = longFormer.get_output(doc_data)
        docs_representations[num]=last_hidden_layer
        print(last_hidden_layer)
    return


def save_tensors_docs(file_name):
    tensor_a = torch.rand(2, 3)
    print(tensor_a)
    tensor_b = torch.rand(1, 3)
    print(tensor_b)
    doc_tensors = {1: tensor_a, 2: tensor_b}
    torch.save(doc_tensors, file_name)

def retrieve_tensors_doc(file_name):
    loaded = torch.load(file_name)
    print(loaded)
    print(loaded[1])
    print(loaded[2])


if __name__ == "__main__":
    eng_directory = '/home/christine/news-commentary/aligned/German-English/English'
    get_doc_representation(eng_directory, 2)

    #save_directory = '/home/christine/news-commentary/aligned/German-English/trail_save'
    #save_tensors_docs(save_directory)
    #retrieve_tensors_doc(save_directory)

