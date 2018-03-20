from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
import numpy as np
#利用TSNE将高维向量降维，便于作图
from sklearn.manifold import TSNE

MIN_LEN = 10

random_state = 170

def load_alexa(filename):
    domain_list = []
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        #row:有两个项，行数+内容
        domain = row[1]
        if len(domain) >=MIN_LEN:
            domain_list.append(domain)
    return domain_list

def load_dga(filename):
    domain_list=[]
    #xsxqeadsbgvpdke.co.uk,Domain used by Cryptolocker - Flashback DGA for 13 Apr 2017,2017-04-13,
    # http://osint.bambenekconsulting.com/manual/cl.txt
    with open(filename) as f:
        for line in f:
            domain=line.split(",")[0]
            if len(domain) >= MIN_LEN:
                domain_list.append(domain)
    return  domain_list

def kmeans_dga():
    print(os.getcwd())
    x1_domain_list = load_alexa("./data/1book/data/top-100.csv")
    x2_domain_list = load_dga("./data/1book/data/dga/dga-cryptolocke-50.txt")
    x3_domain_list = load_dga("./data/1book/data/dga/dga-post-tovar-goz-50.txt")

    x_domain_list = np.concatenate((x1_domain_list, x2_domain_list, x3_domain_list))
    print(x_domain_list)
    y1 = [0]*len(x1_domain_list)
    y2 = [1]*len(x2_domain_list)
    y3 = [1]*len(x3_domain_list)

    y = np.concatenate((y1, y2, y3))
    
    cv = CountVectorizer(ngram_range=[2,2], decode_error="ignore",token_pattern=r"\w", min_df=1)
    x = cv.fit_transform(x_domain_list).toarray()
    print(x)
    model = KMeans(n_clusters=2, random_state=random_state)
    print(x_domain_list)
    y_pred = model.fit_predict(x)

    tsne = TSNE(learning_rate=100)
    x = tsne.fit_transform(x)
    print("jj")
    for i,label in enumerate(x):
        #print label
        x1,x2=x[i]
        if y_pred[i] == 1:
            plt.scatter(x1,x2,marker='o')
        else:
            plt.scatter(x1, x2,marker='x')
        #plt.annotate(label,xy=(x1,x2),xytext=(x1,x2))

    plt.show()



if __name__ == "__main__":
    kmeans_dga()