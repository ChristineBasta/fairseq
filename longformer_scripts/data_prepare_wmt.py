


import os
import logging
import argparse
import torch
from  get_data_statistics import get_files_stats
# to add  number to the beginning of the name
# we treat files as separated docs, we have Europarl, news and TED FILES
# Files are generally divided in English and other language
# English files have '.en' extension and German files have '.de' extension
def numbering_files(eng_dir, other_lng_dir, extension, extension_lang):
    count = 1
    for filename in sorted(os.listdir(eng_dir)):
        #the extension is '.en' or the other language
        if filename.endswith(extension):
            print(count)
            new_file_numbered = str(count) + '_' + filename
            print(new_file_numbered)
            print(other_lng_dir+'/'+filename)
            os.rename(eng_dir+'/'+filename, eng_dir+'/'+new_file_numbered)
            filename_lang = filename.replace(('.'+extension), ('.'+extension_lang))
            new_file_numbered_lang=new_file_numbered.replace(('.'+extension), ('.'+extension_lang))
            os.rename(other_lng_dir+'/'+filename_lang, other_lng_dir+'/'+new_file_numbered_lang)
            print (filename)
            count = count + 1

            continue
        else:
            continue
def get_file_by_num(dir, num, extension):
    for filename in sorted(os.listdir(dir)):
        # the extension is '.en' or the other language
        if filename.endswith(extension):
            if filename.startswith(str(num)+'_'):
                return dir+'/'+filename

        else:
            continue
    return ''

def read_doc_by_num(eng_dir, num):
    file_path= get_file_by_num(eng_dir, num)
    print(file_path)
    doc_string=''
    doc_list = []
    doc_list.append('<S>')
    lines_paragraphs_indices=[]
    with open(file_path) as fp:
        line = fp.readline()
        count = 1

        while line:
            #print("Line {}: {}".format(count, line.strip()))
            line = fp.readline()
            if(count > 3):
                doc_list.append(line.strip())
                doc_string=doc_string +' '+ (line.strip())
                if (line == '<P>'):
                    lines_paragraphs_indices.append(count)

            count += 1
    return doc_string.strip(), doc_list, lines_paragraphs_indices

# the data we have do not have paragraphs separators,
# so we just need to form the string out of the document
def read_doc_by_num(eng_dir, num):
    file_path= get_file_by_num(eng_dir, num)
    print(file_path)
    doc_string=''
    lines_paragraphs_indices=[]
    with open(file_path) as fp:
        line = fp.readline()
        count = 1

        while line:
            #print("Line {}: {}".format(count, line.strip()))
            line = fp.readline()
            if(count > 1):
                doc_string=doc_string +' '+ (line.strip())
                lines_paragraphs_indices.append(count)

            count += 1
    return doc_string.strip(), lines_paragraphs_indices

def get_text_from_file(file_path):
    lines_paragraphs_indices=[]
    doc_string = ''
    with open(file_path) as fp:
        line = fp.readline()
        count = 1

        while line:
            #print("Line {}: {}".format(count, line.strip()))
            line = fp.readline()
            if(count > 1):
                doc_string=doc_string +' '+ (line.strip())
                lines_paragraphs_indices.append(count)

            count += 1
    return doc_string.strip(), lines_paragraphs_indices
def write_all_docs(eng_dir, lang_dir, eng_file_all_name, lang_pair_file_all_name):
    max_file_no=len(os.listdir(eng_dir))
    print(max_file_no)
    sent_doc_dic={}
    sent_num=0
    eng_file_all = open(eng_file_all_name, 'w+')
    lang_pair_file_all = open(lang_pair_file_all_name, 'w+')
    for num in range(1,(max_file_no+1)):
        print(num)
        file_eng_path= get_file_by_num(eng_dir, num)
        print(file_eng_path)
        file_lang_path = get_file_by_num(lang_dir, num)
        print(file_lang_path)
        doc_num=num

        with open(file_eng_path) as fp, open(file_lang_path) as flp:
            line_eng = fp.readline()
            print(line_eng)
            line_lang = flp.readline()
            print(line_lang)

            count_for_file=1
            while line_eng and line_lang:
                #print("Line {}: {}".format(count, line.strip()))
                line_eng = fp.readline().strip()
                line_lang = flp.readline().strip()
                if(count_for_file > 2):
                    if (line_eng != '<P>'):
                        eng_file_all.write(line_eng+'\n')
                        lang_pair_file_all.write(line_lang+'\n')
                        sent_num = sent_num + 1
                        print('********sent_num********')
                        print(sent_num)
                        print('********doc_num********')
                        print(doc_num)
                        sent_doc_dic[sent_num]=doc_num
                count_for_file += 1
                print('********count_per_file********')
                print(count_for_file)
    return sent_doc_dic

def write_all_docs(eng_dir, lang_dir, eng_file_all_name, lang_pair_file_all_name, extension, extension_lang, stats_dic, max_no_lines):
    #todo: get the number of files in a better way
    max_file_no=len(os.listdir(eng_dir))
    print(max_file_no)
    sent_doc_dic={}
    sent_num=0
    eng_file_all = open(eng_file_all_name, 'w+')
    lang_pair_file_all = open(lang_pair_file_all_name, 'w+')
    for num in range(1,(max_file_no+1)):
        print(num)

        file_eng_path= get_file_by_num(eng_dir, num, extension)
        print(file_eng_path)
        file_name=os.path.basename(file_eng_path)
        #check if file has reasonable no. of lines
        if stats_dic[file_name]<max_no_lines:
            file_lang_path = get_file_by_num(lang_dir, num, extension_lang)
            print(file_lang_path)
            doc_num=num
            if(file_eng_path and file_lang_path):
                with open(file_eng_path) as fp, open(file_lang_path) as flp:
                    line_eng = fp.readline()
                    print(line_eng)
                    line_lang = flp.readline()
                    print(line_lang)

                    count_for_file=1
                    while line_eng and line_lang:
                        #print("Line {}: {}".format(count, line.strip()))
                        #starting from second line
                        line_eng = fp.readline().strip()
                        line_lang = flp.readline().strip()

                        if (line_eng != '<P>' and line_eng):
                            eng_file_all.write(line_eng+'\n')
                            lang_pair_file_all.write(line_lang+'\n')
                            sent_num = sent_num + 1
                            print('********sent_num********')
                            print(sent_num)
                            print('********doc_num********')
                            print(doc_num)
                            sent_doc_dic[sent_num]=doc_num
                        count_for_file += 1
                        print('********count_per_file********')
                        print(count_for_file)
    return sent_doc_dic
if __name__ == "__main__":

    #eng_directory = '/home/christine/news-commentary/aligned/German-English/English'
    #lang_directory = '/home/christine/news-commentary/aligned/German-English/German'
    #eng_file_all_name='/home/christine/news-commentary/aligned/German-English/all_data.eng'
    #lang_pair_file_all_name='/home/christine/news-commentary/aligned/German-English/all_data.de'


    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng_dir", help="The english directory")
    parser.add_argument("--lang_dir", help="The other language directory directory")
    parser.add_argument("--extension", help="extension of files which we need to handle")
    parser.add_argument("--extension_lang", help="extension of files which we need to handle in the other language")
    parser.add_argument("--file_sent_doc", help="file of sentence and doc alignment dictionary")
    parser.add_argument("--eng_file_all_name", help="The english file in which english text from all files in directory is written")
    parser.add_argument("--lang_pair_file_all_name", help="The other language file in which english text from all files in directory is written")
    parser.add_argument("--stats_file",
                        help="statitics for the number of lines")
    parser.add_argument("--max_no_lines",
                        help="max no. lines to consider")

    args = parser.parse_args()
    logger.info(args)
    eng_dir = args.eng_dir
    lang_dir=args.lang_dir
    extension=args.extension
    extension_lang=args.extension_lang
    file_sent_doc = args.file_sent_doc
    eng_file_all_name = args.eng_file_all_name
    lang_pair_file_all_name = args.lang_pair_file_all_name
    stats_file=args.stats_file
    max_no_lines=int(args.max_no_lines)

    numbering_files(eng_dir, lang_dir, extension, extension_lang)
    stats_dic=get_files_stats(eng_dir, stats_file)
    sent_doc_dic=write_all_docs(eng_dir, lang_dir, eng_file_all_name, lang_pair_file_all_name, extension, extension_lang,stats_dic, max_no_lines)
    print(sent_doc_dic)
    torch.save(sent_doc_dic, file_sent_doc)













