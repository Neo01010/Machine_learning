#在判断域名的时候，可以采用以下特征：1、元音所占比例 2、去重后字符个数比例 3、jarccard系数 4、HMM系数
import csv
import matplotlib.pyplot as plt

MIN_LEN = 10


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

def load_alexa(filename):
    domain_list=[]
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        domain=row[1]
        if len(domain) >= MIN_LEN:
            domain_list.append(domain)
    return domain_list

def count2string_jarccard_index(a, b):
    x = set(" "+a[0])
    y = set(" "+b[0])
    for i in range(0, len(a)-1):
        x.add(a[i]+a[i+1])
    x.add(a[len(a)-1]+" ")

    for i in range(0, len(b)-1):
        y.add(b[i]+b[i+1])
    y.add(b[len(b)-1]+" ")

    # return 0.0 + len(x|y)/len(x & y)
    return 0.0 +len(x - y)/len(x|y)



def get_jarccard_index(a_list, b_list):
    x = []
    y = []
    for a in a_list:
        j = 0.0
        for b in b_list:
            j+= count2string_jarccard_index(a, b)
        x.append(len(a))
        y.append(j/len(b_list))
    return x, y


def show_jarccard_index():
    x1_domain_list = load_alexa("/home/ren/Desktop/machine_learning/data/1book/data/top-1000.csv")
    x_1, y_1 = get_jarccard_index(x1_domain_list, x1_domain_list)
    x2_domain_list = load_dga("/home/ren/Desktop/machine_learning/data/1book/data/dga-cryptolocke-1000.txt")
    x_2, y_2 = get_jarccard_index(x2_domain_list, x1_domain_list)
    x3_domain_list = load_dga("/home/ren/Desktop/machine_learning/data/1book/data/dga-post-tovar-goz-1000.txt")
    x_3, y_3 = get_jarccard_index(x3_domain_list, x1_domain_list)

    #plt.subplots：生成多个子图,返回值，fig为整个图像对象，ax为子图数组集合
    fig, ax = plt.subplots()
    ax.set_xlabel("Domain Length")
    ax.set_ylabel("JARCCARD index")
    ax.scatter(x_3, y_3,color='b',label="dga_post",marker='o')
    ax.scatter(x_2, y_2,color='g',label="dga_cryp",marker='v')
    ax.scatter(x_1, y_1,color='r',label="alexa",marker='*')

    #添加图例，位置右下方
    ax.legend(loc="lower right")
    plt.show()

    


if __name__ == "__main__":
    show_jarccard_index()