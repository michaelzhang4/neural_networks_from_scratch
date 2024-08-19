import ordinary_least_regression as OLS
import iterative_least_regression as ILS
import numpy as np, cassiopeia as cass

def OLS_example():
    X = [5,7,12,16,20]

    y = [40,120,180,210,240]

    OLS.OLS(X,y)
    
    
def IRLS_example():
    X = [5,7,12,16,20]

    y = [40,120,180,210,240]

    ILS.IRLS(X,y)
    
def binary_data():
    X=np.array([[1,1,1,18],[0,0,0,20],[0,0,0,28],[0,0,0,25],[0,1,1,23],
                [1,1,0,20],[1,1,1,20],[0,0,0,20],[0,1,0,20],[0,1,1,23],
                [1,1,1,23],[1,1,0,23],[0,0,1,20],[1,1,1,1],[1,1,1,13],
                [1,0,1,14],[1,0,0,16],[0,1,1,17],[0,1,1,18],[1,0,1,18],
                [0,0,1,18],[1,1,1,7],[1,1,1,40],[1,1,1,30],[1,1,1,35],
                [1,0,1,32],[0,0,1,40],[0,0,1,60]])
    y=np.array([1,0,0,0,1,
       0,1,0,0,1,
       1,0,1,0,0,
       0,0,0,1,1,
       1,0,0,1,1,
       1,0,0])
    return (X,y)

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
                [2573,525,335,111.5,107],[2366,500,330,105,108.15],[2275,550,340,101.9,90.4],[2333,125,345,120.8,113.9],[2568,125,345,126.9,110.9]])

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