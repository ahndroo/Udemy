import json

data = json.load(open("data.json",'r'))

def translate(word):
    if word in data:
        w = data[word]
        numDefs = len(w)
        for i in range(int(numDefs)):
            print("Definition {}: {}".format(i+1,w[i]))
    else:
        print("The word '{}' doesn't exist.  Double check your shit".format(word))

if __name__ == '__main__':
    word = input("Enter word: ")
    translate(word)
