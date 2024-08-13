
# Numpy for crunching matrix multiplication and matplotlib to view results
import numpy
# Auxiliary helpers for procuring data
import math, random, time, cassiopeia as cass
# For datasets
import sklearn.datasets

import helpers as h

def binary_data():
    X=np.array([[1,1,1,18],[0,0,0,20],[0,0,0,28],[0,0,0,25],[0,1,1,23],
                [1,1,0,20],[1,1,1,20],[0,0,0,20],[0,1,0,20],[0,1,1,23],
                [1,1,1,23],[1,1,0,23],[0,0,1,20],[1,1,1,1],[1,1,1,13],
                [1,0,1,14],[1,0,0,16],[0,1,1,17],[0,1,1,18],[1,0,1,18],
                [0,0,1,18],[1,1,1,7],[1,1,1,40],[1,1,1,30],[1,1,1,35],
                [1,0,1,32],[0,0,1,40],[0,0,1,60]])
    y=[1,0,0,0,1,
       0,1,0,0,1,
       1,0,1,0,0,
       0,0,0,1,1,
       1,0,0,1,1,
       1,0,0]
    return (X,y)
    
class multinomial_logistic_regression:
    # y is array of array of classes with 1 for correct class and 0 if not correct
    # e.g. [[1,0,0],[0,1,0],[0,0,1]]
    def __init__(self, X, y, encoding):
        self.y = y
        self.classes = len(y[0])
        self.w = np.ones((self.classes,len(X[0])))
        self.b = 1
        self.X = X
        self.n = len(X)
        self.iterations = 1000
        self.alpha = 0.01
        self.min_values=[]
        self.max_values=[]
        self.encoding = encoding
        self.scale_data()
   
    def scale_data(self):
        max_values=[]
        min_values=[]
        self.X=self.X.astype(float)
        for col in range(len(self.X[0])):
            max_val = 0
            min_val = 99999
            for row in range(len(self.X)):
                if self.X[row][col] > max_val:
                    max_val = self.X[row][col]
                if self.X[row][col] < min_val:
                    min_val = self.X[row][col]
            max_values.append(max_val)
            min_values.append(min_val)
        scale_factors=[]
        for i in range(len(max_values)):
            count=0
            m = c = max_values[i]-min_values[i]
            while(c>10):
                c%=10
                count+=1
            scale_factors.append(round(m+10**count,-count))
        for col in range(len(self.X[0])):
            for row in range(len(self.X)):
                self.X[row][col]= (self.X[row][col]-min_values[col])/(max_values[col]-min_values[col])

        self.max_values=max_values
        self.min_values=min_values
        
    def softmax(self,X):
        values = []
        total=0
        y_hat=[]
        for i in range(self.classes):
            weights = self.w[i]
            logit = np.dot(weights.T,X)+self.b
            value = math.e**(logit)
            values.append(value)
            total+=value
        for i in range(self.classes):
            y_hat.append(values[i]/total)
        return y_hat

    def train(self):
        for _ in range(self.iterations):
            for x in range(self.n):
                y_hat=self.softmax(self.X[x])
                for c in range(self.classes):
                    for i in range(len(self.w[c])):
                        dw = (y_hat[c]-self.y[x][c])*self.X[x][i]
                        self.w[c][i]-=self.alpha*dw
                        db = (y_hat[c]-self.y[x][c])
                        self.b-=self.alpha*db

    def scale(self,features):
        for i in range(len(features)):
            features[i]=(features[i]-self.min_values[i])/(self.max_values[i]-self.min_values[i])
        return features
    
    def predict(self,X):
        features = self.scale(np.array(X).astype(float))        
        y_hat = self.softmax(features)
        
        return self.encoding[y_hat.index(max(y_hat))]


def multinomial_classification_data_one():
    encoding = ["Skirmisher","Vanguard","Assassin"]
    # League classes: HP, AD, Armor, Atk Spd, Range
    # Akali, Akshan, Diana, Bel'Veth, Fiora, Gwen, Alistar, Amumu, Gragas
    X=np.array([[2623,118,1.17,102.9,125],[2449,103,105.9,1.32,500],[2493,108,104.1,0.97,150],
                [2293,85.5,111.9,0.85,175],[2303,122.1,112.9,1.23,150],[2575,114,127.4,1.07,150],
                [2725,125.75,126.9,0.99,125],[2283,121.6,101,1.11,125],[2595,123.5,123,1.02,125]])
    # Assuassin [0,0,1], Skirmisher [1,0,0], Vanguard [0,1,0]
    y=np.array([[0,0,1],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0]])

    permutation = np.random.permutation(len(X))

    X = X[permutation]

    y = y[permutation]

    return (X,y,encoding)

def multinomial_classification_data_two():
    encoding = ["Fighter","Controller","Mage","Marksman","Slayer","Tank","Specialist"]
    # Garen, Irelia, Olaf, Sett, Illaoi
    # Lulu, Nami, Thresh, Milio, Bard
    # Ryze, Lux, Taliyah, Ahri, Sylas
    # Ashe, Ezreal, Lucian, Kai'Sa, Jhin
    # Katarina, Rengar, Shaco, Tryndamere, Gwen
    # Alistar, Malphite, Rell, Ornn, K'Sante
    # Azir, Teemo, Heimerdinger, Singed, Gangplank
    # HP, Range, MS, AD, AP
    X=np.array([[2356,175,340,145.5,109.4],[2545,200,335,133,115.9],[2668,125,350,147.9,106.4],[2608,125,340,128,121.4],[2611,125,350,153,120],
                [2159,550,330,91.2,109.3],[2056,550,335,106.7,117.4],[2640,450,330,93.4,33],[2056,525,330,102.4,104.2],[2381,500,330,103,119],
                [2753,550,340,109,93.4],[2263,550,330,110.1,109.4],[2318,525,330,114.1,97.9],[2358,550,330,104,100.9],[2768,175,340,112,115.4],
                [2357,600,325,109.15,104.2],[2334,550,325,108.75,103.9],[2341,500,335,109.3,99.4],[2374,525,335,103.2,96.4],[2474,550,330,138.9,103.9],
                [2508,125,335,112.4,107.9],[2358,125,345,119,105.4],[2313,125,345,114,98],[2532,175,345,134,106.1],[2575,150,340,114,127.4],
                [2725,125,330,125.75,126.9],[2412,125,335,130,121.15],[2378,175,330,106,107.4],[2513,175,335,128.5,121.4],[2665,175,330,123.5,121.4],
                [2573,525,335,111.5,107],[2366,500,330,105,108.15],[2275,550,340,101.9,101.9,90.4],[2333,125,345,120.8,113.9],[2568,125,345,126.9,110.9]])

    y=np.array([[1,0,0,0,0,0,0],[1,0,0,0,0,0,0],[1,0,0,0,0,0,0],[1,0,0,0,0,0,0],[1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0],[0,1,0,0,0,0,0],[0,1,0,0,0,0,0],[0,1,0,0,0,0,0],[0,1,0,0,0,0,0],
                [0,0,1,0,0,0,0],[0,0,1,0,0,0,0],[0,0,1,0,0,0,0],[0,0,1,0,0,0,0],[0,0,1,0,0,0,0],
                [0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0],[0,0,0,0,1,0,0],[0,0,0,0,1,0,0],[0,0,0,0,1,0,0],[0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,0,1]])

    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]

    return (X,y,encoding)

def multinomial_classification_data_three():
    encoding = ["Fighter","Support","Mage","Marksman","Assassin","Tank"]
    no_classes = len(encoding)
    champs = cass.get_champions('EUW')
    no_champs = len(champs)
   
    X=np.empty((no_champs, 17))
    y=np.zeros((no_champs,len(encoding)))
    for i in range(no_champs):
        index = encoding.index(champs[i].tags[0])
        y[i][index]=1
        stats = champs[i].stats
        X[i][0]=stats.armor
        X[i][1]=stats.movespeed
        X[i][2]=stats.armor_per_level
        X[i][3]=stats.health_regen
        X[i][4]=stats.health
        X[i][5]=stats.health_per_level
        X[i][6]=stats.health_regen_per_level
        X[i][7]=stats.magic_resist
        X[i][8]=stats.magic_resist_per_level
        X[i][9]=stats.mana
        X[i][10]=stats.mana_per_level
        X[i][11]=stats.mana_regen
        X[i][12]=stats.percent_attack_speed_per_level
        X[i][13]=stats.attack_range
        X[i][14]=stats.attack_damage
        X[i][15]=stats.attack_damage_per_level
        X[i][16]=stats.attack_speed
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]
    return (X,y,encoding)

def multinomial_classification_data_four():
    encoding = ["Fighter","Support","Mage","Marksman","Assassin","Tank"]
    no_classes = len(encoding)
    champs = cass.get_champions('EUW')
    no_champs = len(champs)

    X=np.empty((no_champs, 6))
    y=np.zeros((no_champs,len(encoding)))
    for i in range(no_champs):
        index = encoding.index(champs[i].tags[0])
        y[i][index]=1
        stats = champs[i].stats
        X[i][0]=stats.health
        X[i][1]=stats.attack_range
        X[i][2]=stats.attack_damage
        # Attack speed is wrong/different from wiki
        # X[i][3]=stats.attack_speed
        X[i][3]=stats.armor
        X[i][4]=stats.magic_resist
        X[i][5]=stats.movespeed

    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]

    return (X,y,encoding)

class node:
    def __init__(self,connections, activation):
        self.w = np.ones((connections))
        self.b = 1
        self.activation = activation
        self.init_randomise()

    def init_randomise(self):
        random.seed(time.time())
        for i in range(len(self.w)):
            self.w[i] = random.random()
        self.b = random.random()

    def compute_output(self, X):
        logit = np.dot(self.w.T,X) + self.b
        return (logit,self.apply_activation(logit))

    def apply_activation(self,z):
        if self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        elif self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'softmax':
            return self.softmax(z)
        else:
            return z
    
    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def sigmoid(self, x):
        return 1/(1+math.e**(-x))

    def tanh(self, x):
        c = math.e**(x)
        nc = math.e**(-x)
        return (c-nc)/(c+nc)

    def softmax(self, x):
        y_hat=[]
        logits=[]
        total=0
        for i in range(len(self.classes)):
            logit = np.dot(self.w[i],x.T)+self.b
            exp = math.e**(logit)
            exps.append(exp)
            total += exp
        for i in range(len(self.classes)):
            y_hat.append(exps[i]/total)
        return y_hat.index(max(y_hat))


# Feed forward neural network
class feed_forward_nn:
    def __init__(self, X, y, encoding,hidden_layers):
        self.X = X
        self.y = y
        self.classes = len(encoding) 
        self.encoding = encoding
        # each item in list is a number of nodes per layer
        # e.g. [7, 9] is 2 layers with 7 and 9 nodes respectively
        self.hidden_layers = hidden_layers
        self.nodes = []
        self.create_layers()
        if self.classes < 2:
            print("Error output classes must be 2 or greater")


    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def sigmoid(self, x):
        return 1/(1+math.e**(-x))

    def tanh(self, x):
        c = math.e**(x)
        nc = math.e**(-x)
        return (c-nc)/(c+nc)

    def softmax(self, x):
        y_hat=[]
        logits=[]
        total=0
        for i in range(self.classes):
            logit = np.dot(self.w[i],x.T)+self.b
            exp = math.e**(logit)
            exps.append(exp)
            total += exp
        for i in range(self.classes):
            y_hat.append(exps[i]/total)
        return y_hat

    def create_layers(self):
        for i,layer_num in enumerate(self.hidden_layers):
            layer = []
            for j in range(layer_num):
                if i == 0:
                    layer.append(node(len(self.X[0]),"None"))
                else:
                    layer.append(node(self.hidden_layers[i-1],"None"))
            self.nodes.append(layer)
        output_layer = []
        if self.classes == 2:
            output_layer.append(node(self.hidden_layers[-1],"sigmoid"))
        else:
            for i in range(self.classes):
                output_layer.append(node(self.hidden_layers[-1],"softmax"))
        self.nodes.append(output_layer)

    def train_sig(self):
        for i in range(len(self.X)):

            outputs = self.forward_pass(self.X[i])
            y_hat = self.sigmoid(outputs[-1])
            errors = []
            output_error = (y_hat-self.y[i])*(self.sigmoid(outputs[-1]))*(1-self.sigmoid(outputs[-1]))
            errors.append(output_error)
            # j is layer number
            for j in range(len(self.nodes)-1):
                # inverse layer number and exclude output layer
                layer = len(self.nodes)-1-j
                for k in range(len(self.nodes[layer])):
                    error = (np.dot(self.nodes[layer][k].w, errors[-1]))*(self.sigmoid(outputs[layer]))*(1-self.sigmoid(outputs[layer]))
                    gradient = error * (outputs[layer-1])
                    self.nodes[j][k].w -= self.alpha * gradient



    def train_soft(self):
        for i in range(len(self.X)):
            outputs = self.forward_pass(self.X[i])
            y_hat = self.softmax(np.array([outputs[-1]]))


    def train(self):
        if self.classes==2:
            self.train_sig()
        else:
            self.train_soft()


    def forward_pass(self, inputs):
        outputs = []
        for layer in range(len(self.nodes)):
            output = []
            for nodes in self.nodes[layer]:
                if layer == 0:
                    # (logit, activation)
                    output.append(nodes.compute_output(inputs))
                else:
                    output.append(nodes.compute_output(np.array(outputs[layer-1])))
            outputs.append(output)
        return outputs


        

class CNN:
    def __init__(self):
        # To do
        self.x=5

def CNN_digits_data():
    digits = sklearn.datasets.load_digits()
    y = digits.target
    X = digits.data

a = feed_forward_nn(np.ones((12,4)),np.ones((12)),["output", "rererere"],[8,6])

