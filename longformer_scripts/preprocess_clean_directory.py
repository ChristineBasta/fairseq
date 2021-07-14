import shutil, os
import data_prepare_wmt
from langdetect import detect
import argparse
import logging


def clean_directory(eng_dir, other_lng_dir, en_extension, extension_lang, new_en_folder, new_lang_folder):
    os.mkdir(new_en_folder)
    os.mkdir(new_lang_folder)
    for filename in sorted(os.listdir(other_lng_dir)):
        # the extension is '.en' or the other language
        filename_wt_ext=filename.split(".")[-1]
        text, _ = data_prepare_wmt.get_text_from_file(eng_dir + '/' + filename_wt_ext+'.'+en_extension)
        text_other_lang, _ = data_prepare_wmt.get_text_from_file(other_lng_dir + '/' + filename_wt_ext+'.'+extension_lang)
        if text and text_other_lang:
            lang = detect(text)
            lang_2 = detect(text_other_lang)
            if lang != en_extension or lang_2 != extension_lang:
                print('*************************************')
                print(text)
                print(filename)
                print(lang)
                print('*************************************')
                # do not move the file to the, else move it
            else:
                print('copy to the new folder')
                shutil.copy(eng_dir + '/' + filename_wt_ext+'.'+en_extension, new_en_folder)
                shutil.copy(other_lng_dir + '/' + filename_wt_ext+'.'+extension_lang, new_lang_folder)
        else:
            print(filename + ' has only one line, which is probably title.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng_dir", help="The english directory")
    parser.add_argument("--lang_dir", help="The other language directory directory")
    parser.add_argument("--extension", help="extension of files which we need to handle")
    parser.add_argument("--extension_lang", help="extension of files which we need to handle in the other language")
    parser.add_argument("--eng_new_dir",
                        help="The english new folder to move the correct files to")
    parser.add_argument("--lang_new_dir",
                        help="The other folder to move the correct files to ")

    args = parser.parse_args()
    logger.info(args)
    eng_dir = args.eng_dir
    lang_dir = args.lang_dir
    extension = args.extension
    extension_lang = args.extension_lang
    eng_new_dir = args.eng_new_dir
    lang_new_dir = args.lang_new_dir

    clean_directory(eng_dir, lang_dir, extension, extension_lang, eng_new_dir, lang_new_dir)
