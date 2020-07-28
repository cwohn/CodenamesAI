print("Importing...")
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#Get a dictionary maping words to their embeddings
#embedding_file = "glove.6B.100d.txt"
embedding_file = input("Enter the name of the file containing the word embeddings\n").strip()
print("Loading embeddings...")
embeddings = {}
with open(embedding_file, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings[word] = vector

#Compute the cosine similarity beween two words
def similarity(word_1, word_2):
    return spatial.distance.cosine(embeddings[word_1], embeddings[word_2])

#Compute the cosine similarity beween two word embeddings
def similarity_to_embedding(embedding_1, embedding_2):
    return spatial.distance.cosine(embedding_1, embedding_2)

#Sort words in accending order of their cosine similarity to word
def similar_words(word, words=embeddings.keys()):
    return sorted(words, key=lambda x: similarity(word, x))

#Sort words in accending order of their cosine similarity to embedding
def similar_words_to_embedding(embedding, words=embeddings.keys()):
    return sorted(words, key=lambda x: similarity(embedding, embeddings[x]))

#Return a dictionary mapping each word in the word list to the similarity it has to the target word
def get_similarity_dict(word_list, target_word):
    interpretation = {}
    for word in word_list:
        interpretation[word] = similarity(word, target_word)
    return interpretation
    
    
####Codenames
#A class to manage the Codenames game
#It sores the board, whose turn it is, and other game info
#It contains function to manipulate the game; taking turns, picking codenames, etc.
class Game:
    def __init__(self):
        self.board = {}
        #self.red_score = 0
        #self.blue_score = 0
        self.turn = "red" #or "blue"
        self.winner = None
        self.red_operator = User()
        self.blue_operator= User()
        self.red_spymaster = User()
        self.blue_spymaster = User()
 
    #The given codename is picked by the given team
    #It is removed from the board
    #If it is black, the game ends and the other team wins
    #If there are no red or no blue codenames left, the coresponding team wins
    #If it's color does not match team, the turn changes to the other team
    def pick_codename(self, codename, team):
        color = self.board[codename]
        del self.board[codename]
        #if there are no reds or no blues left, that team wins
        #if color was black, team loses
        #if color!=team, turn ends
        if color == "black":
            self.winner = "red" if team=="blue" else "blue"
            self.turn = ""
            return
        
        #Check to see if whether red or blue has won by picking all of their words
        has_red = False
        has_blue = False
        for curr_codename, curr_color in self.board.items():
            if not has_red and curr_color=="red":
                has_red = True
            if not has_blue and curr_color=="blue":
                has_blue = True

        if not has_red:
            self.winner = "red"
            self.turn = ""

        elif not has_blue:
            self.winner = "blue"
            self.turn = ""
        
        #If the team picked a codename that did not belong to their team, their turn ends
        if color != team:
            self.turn = "red" if self.turn=="blue" else "blue"

    #Take a turn, which consists of:
    #   1. The spymaster gives a clue word
    #   2. The spymaster gives a clue number
    #   3. The operator gives a list of codenames which it intends to guess
    #        Each of these codenames is guessed in order until:
    #           a) the turn ends (e.g. by picking a grey codename)
    #           b) the game ends (e.g. by picking the last red codename)
    #        or c) all the words have been guessed.
    #   4. The turn flips to the other team
    #Return false if the game has ended, else true
    def take_turn(self):
        print_board(self.board, show_values=True)
        print(f"{self.turn}\'s turn.")
        
        #Red's Turn
        if self.turn == "red":
            #Get the clue from the spymaster
            clue_word = self.red_spymaster.get_clue_word(self.board, self.turn)
            clue_num = self.red_spymaster.get_clue_num(self.board, self.turn, clue_word)
            print(f"Clue: {clue_word} {clue_num}\n\n")
            
            #Get the list of words to be guessed from the operator
            to_pick = self.red_operator.guess_codenames(self.board, clue_word, clue_num)
            
            #Guess the words from the operator
            for codename in to_pick:
                print(f"Picked: {codename}, value is: {self.board[codename]}")
                self.pick_codename(codename, self.turn)
                if self.turn!="red":
                    break
            self.turn = "blue"
                
        #Blue's Turn
        else:
            #Get the clue from the spymaster
            clue_word = self.blue_spymaster.get_clue_word(self.board, self.turn)
            clue_num = self.blue_spymaster.get_clue_num(self.board, self.turn, clue_word)
            print(f"Clue: {clue_word} {clue_num}\n\n")
            
            #Get the list of words to be guessed from the operator
            to_pick = self.blue_operator.guess_codenames(self.board, clue_word, clue_num)
            
            #Guess the words from the operator
            for codename in to_pick:
                print(f"Picked: {codename}, value is: {self.board[codename]}")
                self.pick_codename(codename, self.turn)
                if self.turn!="blue":
                    break
            self.turn= "red"
        
        if self.winner is None:
            return True
        #else
        print(f"{self.winner} wins!")
        return False
    
    #Prompt and read input to select the players
    #(e.g. red spymaster and red operator are AI, while blue spymaster and operator use user input)
    def choose_players(self):
        print("Choose a player for the red spymaster: ")
        print("1. AI")
        print("2. User")
        answer = input("Enter \"1\" or \"2\" to choose\n")
        self.red_spymaster = AI() if "1" in answer else User()
        
        print("Choose a player for the red operator: ")
        print("1. AI")
        print("2. User")
        answer = input("Enter \"1\" or \"2\" to choose\n")
        self.red_operator = AI() if "1" in answer else User()

       
        print("Choose a player for the blue spymaster: ")
        print("1. AI")
        print("2. User")
        answer = input("Enter \"1\" or \"2\" to choose\n")
        self.blue_spymaster = AI() if "1" in answer else User()

        print("Choose a player for the blue operator: ")
        print("1. AI")
        print("2. User")
        answer = input("Enter \"1\" or \"2\" to choose\n")
        self.blue_operator = AI() if "1" in answer else User()

    
    #Start a game of Codenames!
    def go(self):
        self.board = get_new_board(get_codenames())
        
        self.choose_players()
        
        while self.take_turn():
            pass
            
#A class to get user input and pass it to the game
class User:
    def __init__(self):
        pass
    
    def get_clue_word(self, board={}, team=""):
        print("Pick a clue for this board:")
        print_board(board, show_values=True)
        print()
        return input("Clue: ")
    
    def get_clue_num(self, board={}, team="", clue=""):
        return int(input("Clue num: "))
    
    def guess_codenames(self, board, clue_word, clue_num):
        print_board(board)
        print("You will enter a list of words to guess, one at a time")
        print("Once you enter the complete list, the words will be guessed in order")
        print("Enter a line containing \"!\" to mark the end of the list")
        words = []
        word = input("Enter the first word in the list\n").lower()
        while "!" not in word:
            while word not in board.keys():
                word = input("\nInvalid codename, please try again\n").lower()
            words.append(word)
            word = input("\nEnter the next word in the list\n")
        return words
    
#A class to get AI decisions and pass them to the game
class AI:
    def __init__(self):
        pass
    
    def get_clue_word(self, board={}, team=""):
        return AI_get_clue_word(board, team)
    
    def get_clue_num(self, board={}, team="", clue=""):
        return AI_get_clue_num(board, team, clue)
    
    def guess_codenames(self, board, clue_word, clue_num):
        return AI_interpret_clue(board.keys(), clue_word)[:clue_num]

#Define a list of codenames (the words on cards that need to be guessed) (there are about 200 of these)
#Only add words that have embeddings
def get_codenames():
    ret = []
    codename_file = "codenames_word_list.txt"
    #codename_file = input("Enter the name of the file containing the word codenames\n").strip()
    with open(codename_file, 'r') as f:
        for line in f:
            if line.strip() in embeddings.keys():
                ret.append(line.strip())
    return ret
            
#A board is a mapping of 25 codenames to their identities
#9 codenames are red (red player gets point if they pick it, blue player's turn ends if they pick it)
#8 codenames are blue (blue player gets point if they pick it, red player's turn ends if they pick it)
#7 codenames are grey (if you pick this, your turn ends)
#1 codename is black (if you pick this you instantly lose)
#In the actual game, a new board is created by selecting at random a pre-made template
#In this implementation, boards are created spontaneously and randomly
#First, randomly select 9 codenames to be red, then 8 to be blue, 7 to be grey, and then the one left is black
def get_new_board(codenames):
    board = {}
    i = 0
    for word in np.random.choice(codenames, size=25, replace=False):
        if i<9:
            board[word] = "red"
        elif i<9+8:
            board[word] = "blue"
        elif i<9+8+7:
            board[word] = "grey"
        else:
            board[word] = "black"
        i+=1
    return board

def print_board(board, show_values=False):
    line = ""
    #Randomize the Order
    for word in np.random.choice(list(board.keys()), len(board.keys()), replace=False):
        if show_values:
            line+=word + f"({board[word]})   "
        else:
            line+=word + "    "
        if line.count(" ")/3 >= 5:
            print(line)
            print()
            line = ""
    print(line)


    
def AI_interpret_clue(codenames, clue_word, clue_num=25):
    return similar_words(clue_word, words=codenames)[:clue_num]

#heuristic function
def AI_evaluate_pick(board, team, codename):
    color = board[codename]
    if color == team:
        return 5
    if color == "red" or color == "blue":
        return -8
    if color == "grey":
        return -2
    if color == "black":
        return -20

#Compute a heuristic score for the expected utility of the given clue
#The score is computed by summing the value of picking each codename times the probability of that codeword being
#P(pick=X | clue=Y) := the probability that codename X will be picked given the clue Y
#h(x) is the value of codename x
#The score of the clue is the sum (over each codename X) of P(pick=X | clue=Y)*h(x)
#P(pick=X | clue=Y) is estimated using similarity
#   Sum the similarty of each codename to the clue
#   The probability of a given word being selected is the similarty of the word over the total similarity
#       (i.e. the percent of the total similarity that this word accounts for)
#   Except, lower similarity score means it is more similar. So, we do total similarity minus similarity of the word
def AI_evaluate_clue(board, team, clue):
    score = 0
    sim_dict = get_similarity_dict(board.keys(), clue)
    total = sum(sim_dict.values())
    for codename in sim_dict.keys():
        prob = (total-sim_dict[codename])/total
        value = AI_evaluate_pick(board, team, codename)
        score += prob*value
    return score
    
#Loop through every word we have embeddings for and evaluate it
#Return the word with the maximum score
def AI_get_clue_word(board, team, max_clue_num=5):

    max_score = float("-inf")
    max_clue_word = ""
    max_clue_num = 0
    words_since_last_change = 0
    for clue in embeddings.keys():
        
        #Optional time-saving code
        #In my testing it seemed like this didn't sacrifice much quality
        #I think that words are sorted in order how how commonly they are used
        #Uncommon words probably make bad clues.
        #
        #If you've scanned 20,000 words since finding a new maximum,
        #then just give up to save time.
        words_since_last_change += 1
        if words_since_last_change > 20000:
            break
        
        #(You can't use codenames as clues)
        if clue in board.keys():
            continue
        
        curr_score = AI_evaluate_clue(board, team, clue)
        
        if curr_score > max_score:
            max_score = curr_score
            max_clue_word = clue
            words_since_last_change = 0
            print(f"~Possible clue: {clue} {max_clue_num}, Matches: {AI_interpret_clue(board, clue, 5)}, score: {max_score}")
  
        
    return max_clue_word

#Return the number of words, that belong to the same team, that are in the top 5 most similar to the clue
def AI_get_clue_num(board, team, clue):
    count = 0
    for codename in similar_words(clue, words=board.keys())[:5]:
        if board[codename] == team:
            count+=1
    return count

##Start a game
print("Starting game...")
Game().go()
