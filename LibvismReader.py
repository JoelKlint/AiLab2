

def readFile(file, langValue):

    out = ''
    for line in file:
        info = line.split('\t')
        out += ('%s 1:%s 2:%s \n' %(langValue, info[0].strip(), info[1].strip()))

    return out

english = open('english.txt', 'r')
eng_res = readFile(english, '1')
english.close()

writer = open('english_formated.txt', 'w')
writer.write(eng_res)
writer.close()

french = open('french.txt', 'r')
fr_res = readFile(french, '0')
french.close()

writer = open('french_formated.txt', 'w')
writer.write(fr_res)
writer.close()

writer = open('total_formated.txt', 'w')
writer.write(eng_res)
writer.write(fr_res)
writer.close()