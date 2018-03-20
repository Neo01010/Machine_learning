#apriori只支持python2.7版本
from apriori import apriori
from apriori import generateRules
import re

if __name__ == "__main__":
    myDat = []
    with  open("./data/ibook/data/xss-2000.txt") as f:
        for line in f:
            index = line.find("?")
            tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29',line)
            myDat.append(tokens)
        

    L, suppData = apriori(myDat, 0.15)
    rules = generateRules(L, suppData, miniConf=0.5)
    print(rules)