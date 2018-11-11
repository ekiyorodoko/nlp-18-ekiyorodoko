#   combine data from different files into one file
data_paths = ["dataset/imdb_labelled.txt","dataset/yelp_labelled.txt", "dataset/amazon_cells_labelled.txt"]

for i in range(len(data_paths)):
    fo = open(data_paths[i], 'r')
    fw = open('dataset.txt', 'a')

    for line in fo:
        fw.write(line)

    fo.close()
    fw.close()


