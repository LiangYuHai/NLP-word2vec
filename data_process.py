import codecs

class Word:
    def __init__(self):
        self.data=self.load_word()
        
    def load_word(self):
        data = []
        filepath ='./data.txt'
        input = codecs.open(filepath, 'r', 'utf-8')
        for line in input.readlines():
            data.append([word for word in line.strip().split(',')[1].split()])
        return data