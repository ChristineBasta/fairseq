import os


def get_files_stats(dir, file_stats):
    file_stats_write = open(file_stats, 'w+')

    for filename in sorted(os.listdir(dir)):
        print(filename)
        total_lines=sum(1 for i in open(dir+'/'+filename, 'rb'))
        file_stats_write.write(filename +'  '+str(total_lines)+ '\n')

if __name__ == "__main__":
    eng_directory = '/home/christine/news-commentary/aligned/German-English/English'
    get_files_stats(eng_directory, '/home/christine/news-commentary/aligned/German-English/eng_stats')