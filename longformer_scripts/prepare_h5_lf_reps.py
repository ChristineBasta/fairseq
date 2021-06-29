import h5py
import torch
import argparse
import logging

#file_train, file_valid, file_test
def prepare_dict(train_h5_file, valid_h5_file, test_h5_file):
    dict_train_valid_test={}
    dict_train = {}
    with h5py.File(train_h5_file, "r") as f_train:
        keys = list(f_train.keys())
        for key in keys:
            print(key)
            data = f_train[str(key)][:]
            tensor_data = torch.tensor(data)

            dict_train[int(key)] = tensor_data
    dict_valid = {}
    with h5py.File(valid_h5_file, "r") as f_valid:
        keys = list(f_valid.keys())
        for key in keys:
            print(key)
            data = f_valid[str(key)][:]
            tensor_data = torch.tensor(data)

            dict_valid[int(key)] = tensor_data
    dict_test = {}
    with h5py.File(test_h5_file, "r") as f_test:
        keys = list(f_test.keys())
        for key in keys:
            print(key)
            data = f_test[str(key)][:]
            tensor_data = torch.tensor(data)

            dict_test[int(key)] = tensor_data

    dict_train_valid_test['train']=dict_train
    dict_train_valid_test['valid']=dict_valid
    dict_train_valid_test['test'] = dict_test
    return dict_train_valid_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_training_path", help="The training dictionary path")
    parser.add_argument("--dict_valid_path", help="The valid dictionary path")
    parser.add_argument("--dict_test_path", help="The test dictionary path")
    parser.add_argument("--file_to_save", help="final dict file")

    args = parser.parse_args()
    logger.info(args)

    train_h5_file = args.dict_training_path
    valid_h5_file = args.dict_valid_path
    test_h5_file = args.dict_test_path
    file_to_save = args.file_to_save

    dict_all = prepare_dict(train_h5_file, valid_h5_file, test_h5_file)
    torch.save(dict_all, file_to_save)