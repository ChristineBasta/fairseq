import torch
from bs4 import BeautifulSoup
import logging
import argparse


def prepare_test_dev(file_path_en_read,file_path_lan_read,file_path_en_write,file_path_lan_write):
    f_en = open(file_path_en_read, 'r')
    f_lan = open(file_path_lan_read, 'r')
    f_write_en=open(file_path_en_write, 'w+')
    f_write_lan= open(file_path_lan_write, 'w+')
    doc_text={}
    sen_doc_align={}
    #english
    data = f_en.read()
    soup = BeautifulSoup(data, features="lxml")
    contents_en = soup.findAll('doc')

    #other lang
    data_lan = f_lan.read()
    soup_lan = BeautifulSoup(data_lan,  features="lxml")
    contents_lan = soup.findAll('doc')

    doc_num=1
    seg_num = 1
    for content in contents_en:

        #print(doc_num)
        #getting doc id as this is the main ref in the other language file
        docid = content['docid']
        #print('DOC ID:')
        #print(docid)
        segments = content('seg')
        #check if we should remove the \' or not
        #first sentence does not have am end fullstop so replacing \n with fullstop and then in case of two fullstops replace one
        doc_text_content=content.text.strip().replace('\n', '. ').replace('\\\'', '\'')
        doc_text[doc_num] =doc_text_content.replace('..', '.')

        doc_lang=soup_lan.find(docid=docid)
        segments_lang=doc_lang('seg')
        #print('doc lang by id :')
        #print(doc_lang)
        '''
        print('segments')
        '''

        for seg in segments:
            #print(seg['id'])
            if(seg.text!='\n'):
                #print(seg.text)
                f_write_en.write(seg.text+'\n')
                sen_doc_align[seg_num]=doc_num
                seg_num+=1

        for seg in segments_lang:
            #print(seg['id'])
            if(seg.text!='\n'):
                #print(seg.text)
                f_write_lan.write(seg.text+'\n')

        doc_num+=1
    return doc_text, sen_doc_align

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_read_src", help="src file, should be english")
    parser.add_argument("--file_read_ref", help="ref file")
    parser.add_argument("--file_write_src", help="written src file")
    parser.add_argument("--file_write_ref", help="written ref file")
    parser.add_argument("--doc_text_save", help="text of docs to get the doc representations")
    parser.add_argument("--sen_doc_align", help="sen and doc alignment file to send to the code")

    args = parser.parse_args()
    logger.info(args)
    file_read_src = args.file_read_src
    file_read_ref = args.file_read_ref
    file_write_src = args.file_write_src
    file_write_ref = args.file_write_ref
    doc_text_save_file = args.doc_text_save
    sen_doc_align_file = args.sen_doc_align

    doc_text, sen_doc_align = prepare_test_dev(file_read_src,file_read_ref, file_write_src, file_write_ref)
    torch.save(doc_text, doc_text_save_file)
    torch.save(sen_doc_align, sen_doc_align_file)