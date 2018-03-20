

def createC1(dataSet):
    c1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
    c1.sort()
    #map对c1中的每一个数据进行frozenset运算：返回一个冻结的集合，该集合不能添加和删除数据
    print(map(frozenset, c1))
    return map(frozenset, c1)

def scanD(D, Ck, minSupport):
    ssCnt = {}
    DL = []
    for tid in D:
        print(tid,"TID",Ck)
        DL.append(tid)
        for can in Ck:
            print(can,"CAN")
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1
    print(D)
    numItems = float(len(list(DL)))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
            supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    print("jjj")
    print(lenLk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k - 2];
            L1.sort();
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D =map(set, dataSet)
    #L1:1项集  suppData:dict，1项集：支持度
    L1, suppData = scanD(D, C1, minSupport)
    L= [L1]
    K = 2

    while len(L[K - 2]) > 0:
        Ck = aprioriGen(L[K - 2], K)
        Lk, supK = scanD(D, Ck, minSupport)


if __name__ == "__main__":
    myData = [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]
    
    #minConf:支持度设置为0.5，置信度设置为0.7
    L, suppData = apriori(myData, 0.5)
    rules = generateRules(L, suppData, minConf=0.7)
