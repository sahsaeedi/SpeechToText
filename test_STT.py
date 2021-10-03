from STT import SpeechToText


obj = SpeechToText('./40_AzariJahromi.wav')
print('Start Process ...')
sentence = obj.predict()
with open('test.txt', 'w', encoding= 'utf8') as f:
    f.write(sentence)
print('End Process')