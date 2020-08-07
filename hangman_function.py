#f=open("words.txt","r")
#words = f.read().splitlines()

def hangman(input_word,words):
	i = 0
	for word in words:
		if len(word) < 3:
			del words[i]
		i += 1

	wlength=len(input_word)


	i = len(words)
	for word in words[::-1]:
		i -= 1
		if len(word) != wlength:
			del words[i]
	input_word_list=[]
	for i in input_word:
		input_word_list.append(i)

	for index,item in enumerate(input_word_list):
		if item!='_':
			i = len(words)
			for word in words[::-1]:
				i -= 1
				if word[index]!=item:
					del words[i]
	return words
	
