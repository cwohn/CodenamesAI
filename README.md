# CodenamesAI
This project is an implementation of the board games "Codenames" and an AI that plays the game based on vector word embeddings.
The game's rules can be found here: https://czechgames.com/files/rules/codenames-rules-en.pdf
I used GloVe word embeddings which can be downloaded here: https://nlp.stanford.edu/projects/glove/

The AI works by computing the cosine similarity of word vectors. If given a clue word and a clue number, it sorts the codenames by ascending order of cosine similarity to the clue word and picks the first N words (where N is the clue number). 
If the AI is trying to come up with clues, it goes through every word in its dictionary and computes the cosine similarity of the possible clue to each codename. This similarity is converted to a probability of the operator guessing the codename given the clue and that probability is multiplied by a heuristic value of picking the word. 

When I tested the AI, it was very slow -- slower even than humans. The AI spymaster took around 10 minutes to give a clue at the start of a game, and an AI-only game lasted around 45 minutes. I found that most of the time spent searching after the first few minutes did not yield results, so I implemented a search cut-off: If the AI had examined 20,000 words in a row without updating the "best" clue so far, it ended its search. I don't believe this dramatically impacted the quality of the AI's clues, though I have no qualitative way to test this without conducting experiments involving human participants.

The AI is not especially good. Although it is better than some humans I've played Codenames with, most humans are significantly better. Its clues are often baffling, but sometimes they are surprisingly solid, even clever. While it might be possible to enforce these rules by using grammatical formulae or storing conjugations/plurals of every word, I don't think it's worth the effort, as most humans can find very close synomnmys for most words.

The rules about what codenames are allowed are not strongly enforced as most of them are difficult to specify; for instance, clues based on rhymes are not allowed by the rules. Also, while the AI can't give a clue that is exactly identical to one of the codenames, it can give plurals or different conjugations of codenames, which it shouldn't be allowed to do. 

There are other ways (perhaps more sophisticated) ways to implement AI in Codenames. Theoretically, an Expectiminimax algorithm could be used: Each clue is an action that could lead to one of several new game states according to a probability distribution, eventually reaching a terminal state with a definite value. However, the branching factor is far too large for this to be computationally feasible. 


Sample Output:

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Importing...
Enter the name of the file containing the word embeddings
glove.6B.100d.txt
Loading embeddings...
Starting game...
Choose a player for the red spymaster: 
1. AI
2. User
Enter "1" or "2" to choose
1
Choose a player for the red operator: 
1. AI
2. User
Enter "1" or "2" to choose
1
Choose a player for the blue spymaster: 
1. AI
2. User
Enter "1" or "2" to choose
1
Choose a player for the blue operator: 
1. AI
2. User
Enter "1" or "2" to choose
1
kangaroo(grey)   capital(grey)   stadium(red)   switch(red)   human(red)   

agent(grey)   hospital(grey)   note(blue)   back(blue)   cricket(black)   

kiwi(grey)   knight(blue)   eye(blue)   time(blue)   robot(red)   

snowman(red)   park(grey)   lap(red)   millionaire(red)   root(red)   

whip(grey)   washer(blue)   jam(red)   india(blue)   green(blue)   


red's turn.
~Possible clue: the 0, Matches: ['time', 'back', 'human', 'green', 'capital'], score: -51.39125485158727
~Possible clue: of 0, Matches: ['time', 'human', 'capital', 'back', 'green'], score: -51.35479837804955
~Possible clue: to 0, Matches: ['time', 'back', 'capital', 'human', 'switch'], score: -51.336039759222025
~Possible clue: a 0, Matches: ['time', 'back', 'green', 'human', 'note'], score: -51.26911747669148
~Possible clue: - 0, Matches: ['time', 'back', 'eye', 'green', 'capital'], score: -51.26429230482186
~Possible clue: ( 0, Matches: ['time', 'back', 'green', 'note', 'park'], score: -51.26139033577479
~Possible clue: '' 0, Matches: ['time', 'back', 'note', 'human', 'green'], score: -51.20633582148073
~Possible clue: or 0, Matches: ['time', 'back', 'note', 'switch', 'human'], score: -51.1625065194226
~Possible clue: $ 0, Matches: ['time', 'back', 'note', 'capital', 'park'], score: -51.1373983637707
~Possible clue: million 0, Matches: ['time', 'capital', 'back', 'human', 'stadium'], score: -51.076447116987545
~Possible clue: billion 0, Matches: ['capital', 'time', 'back', 'note', 'india'], score: -51.07127642037708
~Possible clue: family 0, Matches: ['time', 'back', 'human', 'hospital', 'park'], score: -51.040344841925716
~Possible clue: power 0, Matches: ['time', 'back', 'switch', 'capital', 'human'], score: -51.02338075219609
~Possible clue: system 0, Matches: ['switch', 'time', 'human', 'back', 'capital'], score: -50.95431554708072
~Possible clue: nuclear 0, Matches: ['india', 'human', 'time', 'back', 'agent'], score: -50.943208060966874
~Possible clue: station 0, Matches: ['park', 'hospital', 'time', 'switch', 'stadium'], score: -50.92282224856862
~Possible clue: album 0, Matches: ['jam', 'time', 'back', 'note', 'green'], score: -50.900215083431334
~Possible clue: live 0, Matches: ['time', 'back', 'human', 'jam', 'park'], score: -50.88472920617315
~Possible clue: network 0, Matches: ['time', 'switch', 'root', 'human', 'jam'], score: -50.87018114830441
~Possible clue: computer 0, Matches: ['robot', 'time', 'switch', 'human', 'back'], score: -50.84614414243658
~Possible clue: video 0, Matches: ['time', 'jam', 'robot', 'back', 'switch'], score: -50.82363549897954
~Possible clue: plant 0, Matches: ['green', 'park', 'root', 'human', 'capital'], score: -50.76277191536815
~Possible clue: cell 0, Matches: ['human', 'switch', 'hospital', 'time', 'root'], score: -50.690120078726885
~Possible clue: cuban 0, Matches: ['human', 'time', 'back', 'agent', 'millionaire'], score: -50.61055425906109
~Possible clue: monster 0, Matches: ['robot', 'snowman', 'green', 'jam', 'park'], score: -50.580131735881245
~Possible clue: alien 0, Matches: ['robot', 'human', 'agent', 'millionaire', 'eye'], score: -50.57910736132212
~Possible clue: portable 0, Matches: ['robot', 'switch', 'jam', 'stadium', 'washer'], score: -50.57611666920255
~Possible clue: processor 0, Matches: ['switch', 'root', 'jam', 'robot', 'note'], score: -50.50739666601825
~Possible clue: robots 0, Matches: ['robot', 'human', 'switch', 'snowman', 'eye'], score: -50.39516216337677
~Possible clue: kernel 0, Matches: ['root', 'jam', 'switch', 'robot', 'snowman'], score: -50.34187803823043
Clue: kernel 5


Picked: root, value is: red
Picked: jam, value is: red
Picked: switch, value is: red
Picked: robot, value is: red
Picked: snowman, value is: red
india(blue)   agent(grey)   green(blue)   eye(blue)   capital(grey)   

time(blue)   washer(blue)   hospital(grey)   kangaroo(grey)   lap(red)   

stadium(red)   human(red)   park(grey)   millionaire(red)   note(blue)   

kiwi(grey)   whip(grey)   cricket(black)   back(blue)   knight(blue)   


blue's turn.
~Possible clue: the 0, Matches: ['time', 'back', 'human', 'green', 'capital'], score: -24.167431914701773
~Possible clue: , 0, Matches: ['time', 'back', 'capital', 'green', 'park'], score: -24.03870945516122
~Possible clue: from 0, Matches: ['back', 'time', 'capital', 'human', 'note'], score: -24.034341008626875
~Possible clue: or 0, Matches: ['time', 'back', 'note', 'human', 'green'], score: -23.89524171188345
Clue: or 4


Picked: time, value is: blue
Picked: back, value is: blue
Picked: note, value is: blue
Picked: human, value is: red
stadium(red)   millionaire(red)   india(blue)   cricket(black)   kiwi(grey)   

park(grey)   agent(grey)   lap(red)   capital(grey)   green(blue)   

kangaroo(grey)   whip(grey)   knight(blue)   eye(blue)   hospital(grey)   

washer(blue)   
red's turn.
~Possible clue: the 0, Matches: ['green', 'capital', 'park', 'india', 'eye'], score: -55.427209413488725
~Possible clue: to 0, Matches: ['capital', 'india', 'park', 'green', 'hospital'], score: -55.422169717656146
~Possible clue: a 0, Matches: ['green', 'capital', 'hospital', 'park', 'eye'], score: -55.332843243043335
~Possible clue: at 0, Matches: ['park', 'capital', 'hospital', 'stadium', 'green'], score: -55.264946148784375
~Possible clue: percent 0, Matches: ['capital', 'india', 'green', 'hospital', 'lap'], score: -55.16626941104232
~Possible clue: million 0, Matches: ['capital', 'stadium', 'park', 'india', 'hospital'], score: -55.15192847952068
~Possible clue: center 0, Matches: ['park', 'hospital', 'capital', 'stadium', 'green'], score: -55.145754135314164
~Possible clue: station 0, Matches: ['park', 'hospital', 'stadium', 'capital', 'green'], score: -55.120815164996344
~Possible clue: car 0, Matches: ['lap', 'green', 'capital', 'park', 'hospital'], score: -55.04343656420136
~Possible clue: race 0, Matches: ['lap', 'green', 'park', 'stadium', 'whip'], score: -54.990141368146524
~Possible clue: san 0, Matches: ['park', 'stadium', 'capital', 'hospital', 'green'], score: -54.953688766609005
~Possible clue: francisco 0, Matches: ['park', 'agent', 'stadium', 'hospital', 'capital'], score: -54.9262593699124
~Possible clue: seconds 0, Matches: ['lap', 'green', 'stadium', 'eye', 'park'], score: -54.90844922178466
~Possible clue: miami 0, Matches: ['stadium', 'agent', 'hospital', 'green', 'park'], score: -54.83500195485699
~Possible clue: las 0, Matches: ['park', 'stadium', 'capital', 'millionaire', 'hospital'], score: -54.81404767851212
~Possible clue: vegas 0, Matches: ['park', 'millionaire', 'stadium', 'capital', 'agent'], score: -54.77582624708651
~Possible clue: burbank 0, Matches: ['park', 'hospital', 'stadium', 'capital', 'agent'], score: -54.734572984927645
~Possible clue: andretti 0, Matches: ['lap', 'millionaire', 'green', 'kiwi', 'knight'], score: -54.6917260575583
~Possible clue: gaylord 0, Matches: ['millionaire', 'park', 'stadium', 'whip', 'lap'], score: -54.68587743231993
Clue: gaylord 3


Picked: millionaire, value is: red
Picked: park, value is: grey
kangaroo(grey)   green(blue)   hospital(grey)   knight(blue)   agent(grey)   

india(blue)   stadium(red)   cricket(black)   whip(grey)   eye(blue)   

capital(grey)   kiwi(grey)   washer(blue)   lap(red)   
blue's turn.
~Possible clue: the 0, Matches: ['green', 'capital', 'india', 'eye', 'stadium'], score: -21.271316044150094
~Possible clue: , 0, Matches: ['capital', 'green', 'india', 'hospital', 'eye'], score: -21.040207433603385
~Possible clue: '' 0, Matches: ['green', 'agent', 'eye', 'knight', 'capital'], score: -21.027594124365564
~Possible clue: or 0, Matches: ['green', 'eye', 'capital', 'agent', 'hospital'], score: -20.84796117635239
~Possible clue: white 0, Matches: ['green', 'knight', 'eye', 'agent', 'whip'], score: -20.788974238556488
~Possible clue: black 0, Matches: ['green', 'knight', 'eye', 'capital', 'agent'], score: -20.774001462966716
~Possible clue: red 0, Matches: ['green', 'eye', 'knight', 'agent', 'stadium'], score: -20.759782345864256
~Possible clue: blue 0, Matches: ['green', 'eye', 'knight', 'kiwi', 'kangaroo'], score: -20.680967707298503
~Possible clue: dark 0, Matches: ['green', 'eye', 'knight', 'agent', 'capital'], score: -20.64381275410039
~Possible clue: skin 0, Matches: ['eye', 'green', 'hospital', 'kangaroo', 'washer'], score: -20.521816528659546
Clue: skin 3


Picked: eye, value is: blue
Picked: green, value is: blue
Picked: hospital, value is: grey
kangaroo(grey)   india(blue)   washer(blue)   knight(blue)   kiwi(grey)   

stadium(red)   cricket(black)   lap(red)   whip(grey)   capital(grey)   

agent(grey)   
red's turn.
~Possible clue: the 0, Matches: ['capital', 'india', 'stadium', 'agent', 'knight'], score: -39.835546463749
~Possible clue: on 0, Matches: ['capital', 'india', 'stadium', 'agent', 'lap'], score: -39.81129433213634
~Possible clue: at 0, Matches: ['capital', 'stadium', 'india', 'lap', 'agent'], score: -39.71434930500518
~Possible clue: second 0, Matches: ['lap', 'india', 'capital', 'stadium', 'knight'], score: -39.69623477122475
~Possible clue: down 0, Matches: ['capital', 'lap', 'india', 'knight', 'stadium'], score: -39.64923888553308
~Possible clue: off 0, Matches: ['capital', 'lap', 'india', 'stadium', 'agent'], score: -39.64619987709703
~Possible clue: points 0, Matches: ['lap', 'knight', 'stadium', 'capital', 'agent'], score: -39.61875700614296
~Possible clue: lead 0, Matches: ['lap', 'capital', 'india', 'agent', 'stadium'], score: -39.58509952616037
~Possible clue: behind 0, Matches: ['lap', 'stadium', 'india', 'capital', 'agent'], score: -39.53550736851973
~Possible clue: race 0, Matches: ['lap', 'stadium', 'whip', 'cricket', 'india'], score: -39.51712293749375
~Possible clue: san 0, Matches: ['stadium', 'capital', 'agent', 'knight', 'lap'], score: -39.46243618003312
~Possible clue: floor 0, Matches: ['lap', 'stadium', 'whip', 'capital', 'agent'], score: -39.45961008503641
~Possible clue: seconds 0, Matches: ['lap', 'stadium', 'whip', 'knight', 'capital'], score: -39.23847278341232
~Possible clue: laps 0, Matches: ['lap', 'stadium', 'whip', 'knight', 'cricket'], score: -39.13650230753898
Clue: laps 2


Picked: lap, value is: red
Picked: stadium, value is: red
red wins!

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



