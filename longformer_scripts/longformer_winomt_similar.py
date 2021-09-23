import torch
from bs4 import BeautifulSoup
import logging
import argparse

from doc_representations import DocRepresent
import h5py
import prepare_h5_lf_reps

doc_represent=DocRepresent()
def get_representation( line, kind_reps):
    print(kind_reps)
    representation=None
    if (kind_reps == 1 or kind_reps == 2):
        print('entered')
        representation = doc_represent.get_representations_classify(line, kind_reps)
    elif (kind_reps == 3 or kind_reps == 4):
        representation = doc_represent.get_representations_model(line, kind_reps)
    elif kind_reps == 5:
        representation = doc_represent.get_certain_tokens(line, kind_reps, classify_model=True)
    elif kind_reps == 6 or kind_reps == 7:
        representation = doc_represent.get_certain_tokens(line, kind_reps, classify_model=False)
    elif (kind_reps == 8 or kind_reps == 9 or kind_reps == 10 or kind_reps == 11):
        representation = doc_represent.get_representations_led(line, kind_reps)

    return representation

def prepare_translation_dictionaries(file_path, file_h5_name, kind_reps):
    print(file_path)
    count = 1
    saving_file = h5py.File(file_h5_name, 'a')
    sent_doc_dic={}
    with open(file_path) as fp:
        # do not write first line
        line = fp.readline()
        print(line)
        lf_line_rep=get_representation(line, kind_reps)

        representation_numpy = lf_line_rep.cpu().data.numpy()
        saving_file.create_dataset(str(count), data=representation_numpy)
        sent_doc_dic[count] = count
        count += 1
        while line:
            print(count)
            # print("Line {}: {}".format(count, line.strip()))
            line = fp.readline()
            lf_line_rep=get_representation(line, kind_reps)
            representation_numpy = lf_line_rep.cpu().data.numpy()
            saving_file.create_dataset(str(count), data=representation_numpy)
            sent_doc_dic[count] = count
            count += 1
    saving_file.close()
    return sent_doc_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--winomt_file", help="winomt file")
    parser.add_argument("--longformer_h5_file", help="long former dict file ")
    parser.add_argument("--longformer_final_file", help="long former final representations file ")
    parser.add_argument("--sen_doc_alignment", help="sen-doc alignment file")
    parser.add_argument("--kind_reps", help="kind of representations")

    args = parser.parse_args()
    logger.info(args)

    winomt_file = args.winomt_file
    longformer_h5_file = args.longformer_h5_file
    longformer_final_file = args.longformer_final_file
    sen_doc_alignment_file = args.sen_doc_alignment
    kind_reps = int(args.kind_reps)


    sen_doc_align = prepare_translation_dictionaries(winomt_file, longformer_h5_file, kind_reps)
    print(sen_doc_align)


    dict_all = {}
    dict_all['train'] = sen_doc_align
    dict_all['valid'] = sen_doc_align
    dict_all['test'] = sen_doc_align

    torch.save(dict_all, sen_doc_alignment_file)


    train_h5_file = longformer_h5_file
    valid_h5_file = longformer_h5_file
    test_h5_file = longformer_h5_file

    file_to_save = args.longformer_final_file

    dict_all = prepare_h5_lf_reps.prepare_dict(train_h5_file, valid_h5_file, test_h5_file)
    torch.save(dict_all, file_to_save)
