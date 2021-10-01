import csv

def read_google_dataset(path, src_file, ref_file):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=",")
    src_file_to_write = open(src_file, "w+")
    ref_file_to_write = open(ref_file, "w+")

    print(read_tsv)
    count=1
    for row in read_tsv:
        print(row[4])
        print(row[5])
        #skip writing first line
        if(count>1):
            src_file_to_write.write(row[4]+"\n")
            ref_file_to_write.write(row[5]+"\n")
        count+=1
    src_file_to_write.close()
    ref_file_to_write.close()

src_file='/home/christine/google_dataset/en-es.en'
ref_file='/home/christine/google_dataset/en-es.es'
read_google_dataset('/home/christine/google_dataset/Biographies_EN_ES.csv',src_file,ref_file)