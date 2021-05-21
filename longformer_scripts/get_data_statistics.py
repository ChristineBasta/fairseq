import os
import logging
import argparse


def get_files_stats(dir, file_stats):
    file_stats_write = open(file_stats, 'w+')
    stats_dic={}
    for filename in sorted(os.listdir(dir)):
        print(filename.split('_')[0])
        name_by_num=filename.split('_')[0]
        total_lines=sum(1 for i in open(dir+'/'+filename, 'rb'))
        file_stats_write.write(filename +'  '+str(total_lines)+ '\n')
        stats_dic[name_by_num]=total_lines
    return stats_dic

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    logger = logging.getLogger('context.log')  # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng_dir", help="The english directory")
    parser.add_argument("--stats_file", help="statistics file to write the stats")


    args = parser.parse_args()
    logger.info(args)
    print(args)
    eng_directory = args.eng_dir
    stats_file= args.stats_file

    #eng_directory = '/home/christine/news-commentary/aligned/German-English/English'
    #stats_file= '/home/christine/news-commentary/aligned/German-English/eng_stats'
    get_files_stats(eng_directory, stats_file)