import torch
import argparse
import logging

if __name__ == "__main__":

    #eng_directory = '/home/christine/news-commentary/aligned/German-English/English'
    #lang_directory = '/home/christine/news-commentary/aligned/German-English/German'
    #eng_file_all_name='/home/christine/news-commentary/aligned/German-English/all_data.eng'
    #lang_pair_file_all_name='/home/christine/news-commentary/aligned/German-English/all_data.de'


    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_training_path", help="The training dictionary path")
    parser.add_argument("--dict_valid_path", help="The valid dictionary path")
    parser.add_argument("--dict_test_path", help="The test dictionary path")
    parser.add_argument("--dict_sent_doc_all_path", help="The dictionary path of sent-doc alignment of all sets")

    args = parser.parse_args()
    logger.info(args)
    dict_training_path = args.dict_training_path
    dict_valid_path = args.dict_valid_path
    dict_test_path = args.dict_test_path
    dict_sent_doc_all_path = args.dict_sent_doc_all_path

    dict_training=torch.load(dict_training_path)
    dict_valid=torch.load(dict_valid_path)
    dict_test=torch.load(dict_test_path)

    dict_all = {}
    dict_all['train'] = dict_training
    dict_all['valid'] = dict_valid
    dict_all['test'] = dict_test

    torch.save(dict_all, dict_sent_doc_all_path)
