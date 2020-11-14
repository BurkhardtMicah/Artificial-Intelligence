import itertools
import math
from matplotlib import style    
import matplotlib.pyplot as plt 
import numpy as np
import time
##############################################################
#graph sets of xy coordinates
def graph_coords(x2, y2, min_dist, generation):
    #Define graph style
    style.use('dark_background')
    plt.clf()
    # plotting the points
    plt.plot(x2, y2,'yo-', label="Optimum Path")
    for i in range(len(x2) - 1):
        plt.annotate(i + 1, (x2[i], y2[i]), textcoords="offset points", xytext=(0,5), ha = 'center')
        
    # naming the axes 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.legend()
    # giving a title to my graph 
    plt.title(("Optimum Distance : " + str(round(min_dist, 2)) + " Gen: " + str(generation)))
      
    # function to show the plot 
    plt.pause(.05)
    plt.show()    

    return

########################################
class node():
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
########################################
class chromosome():
    def __init__(self, name, nodelist):
        self.name = name
        self.genes = nodelist
        self.distance = math.inf
        
    def listNames(self):
        names = []
        for node in self.genes:
            names.append(node.name)
        return names
            
########################################
#read and parse data file
def read_datafile(path):
    #Import data file
    i = 0
    x = []
    y = []
    with open((path), "r") as file:
      for line in file:
        split_line = line.strip().split(" ")
        
        #Track line number to remove header info
        if i > 6:
            #Populate x,y coordinate pairs into arrays
            x.append(float(split_line[1]))
            y.append(float(split_line[2]))
        #increment line counter
        i += 1    
    return x, y
#########################################
#Calculate distance for the trip
def calculate_trip_dist(trip):
    dist = 0
    for j in range(len(trip)-1):
        node1 = nodes[trip[j]   -1]
        node2 = nodes[trip[j+1] -1]
        
        dist = dist + (math.hypot(node1.x - node2.x, node1.y - node2.y))
    
    #add distance back to first node
    node1 = nodes[trip[0]  -1] #first node
    node2 = nodes[trip[-1] -1] #last node   
    dist = dist + (math.hypot(node1.x - node2.x, node1.y - node2.y))
    
    return dist    
#########################################
def getNumMating(gaPopSize, matingPercentage):
    numMatingChromosomes = int(matingPercentage * gaPopSize)
    if(numMatingChromosomes % 2 == 1 and numMatingChromosomes < gaPopSize):
        numMatingChromosomes += 1
    elif(numMatingChromosomes % 2 == 1 and numMatingChromosomes >= gaPopSize):
        numMatingChromosomes -= 1
    elif(numMatingChromosomes == 0):
        numMatingChromosomes = 2   
    return numMatingChromosomes
    #else even pairs - do nothing
#########################################
def sortChromosomes(populationArr):
    outputPopArr = []
    for chromo in populationArr:
        dist = chromo.distance
        #print(dist)
        if(len(outputPopArr) == 0):
            outputPopArr.append(chromo)
            #print("output initialized")
        else:
            i = 0
            added = False
            for outChromo in outputPopArr:
                if(outChromo.distance >= dist):
                    #print(str(outChromo.distance) + " > " + str(dist)) 
                    outputPopArr.insert(i, chromo)
                    added = True
                    break
                else:
                    #if chromosome is larger than all current, place at end
                    i += 1
            if(not added):
                outputPopArr.append(chromo)
    return outputPopArr
##############################################################
def getXY(trip):
    x = []
    y = []
    for gene in trip:
        x.append(nodes[gene -1].x)
        y.append(nodes[gene -1].y)
                      
    return x, y
##############################################################
def pairMates(matingChromosomes):
    used = []
    pairedMates = []
    i = 1
    j = numMatingChromosomes - 1
    if(i == j):
        topPerformersMate = 1
    else:
        topPerformersMate = np.random.randint(1,j)  
    for pair in itertools.combinations(matingChromosomes,2):
        c1 = pair[0].name
        c2 = pair[1].name
        
        if(i == topPerformersMate):
            used.append(c1)
            used.append(c2) 
            pairedMates.append(pair)
            i = i+1
            continue
        elif(not(used.__contains__(c1) or used.__contains__(c2)) and i > topPerformersMate):
            used.append(c1)
            used.append(c2)
            pairedMates.append(pair)
        i = i+1   
    return pairedMates
##############################################################
def generateSplitPoint():
    a = np.random.randint(1,98)
    b = a
    while(a == b):
        b = np.random.randint(1,98)   
    
    splitPoint1 = min(a, b)
    splitPoint2 = max(a, b) 
    return splitPoint1, splitPoint2
##############################################################
def Crossover(parent1genes, parent2genes, splitPoint1, splitPoint2):
    tempChromosome = []
    #First section
    for x in range(splitPoint1):
        tempChromosome.append(nodes[parent1genes[x] - 1])
    #Last Section    
    for x in range(splitPoint2, 100):
        tempChromosome.append(nodes[parent1genes[x] - 1]) 
    #Mid Section
    for gene in parent2genes:
        found = False
        for gene2 in tempChromosome:
            if(gene2.name == nodes[gene -1].name):
                found = True
                break
        if(not found):
            tempChromosome.insert(splitPoint1,(nodes[gene -1]))
            
    return tempChromosome
##############################################################
def deviation(populationArr):
    best = populationArr[0].distance
    devArr = []
    for chromo in populationArr:
        devArr.append(chromo.distance - best)
    
    dev = np.sum(devArr)/ len(devArr)
    
    return dev
##############################################################
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]
##############################################################
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y    
##############################################################
def findOverlap(chromo):
    overlappingI = []

    for i in range(len(chromo.listNames())):
        #i 0-99
        for j in range(len(chromo.listNames())):
            #print(i)
            if(j != i and j != i+1 and j!= i-1 and i != 99 and j != 99):
                #First edge
                A = [chromo.genes[i].x, chromo.genes[i].y]
                B = [chromo.genes[i + 1].x, chromo.genes[i + 1].y]
                #Second edge
                C = [chromo.genes[j].x, chromo.genes[j].y]
                D = [chromo.genes[j + 1].x, chromo.genes[j + 1].y]
                intersection = (line_intersection((A, B), (C, D)))
    
                #check if intersection is on edge line segment
                if(((A[0] >= intersection[0] >= B[0]) or (A[0] <= intersection[0] <= B[0])) and ((A[1] <= intersection[1] <= B[1]) or (A[1] >= intersection[1] >= B[1])) and ((C[0] >= intersection[0] >= D[0]) or (C[0] <= intersection[0] <= D[0])) and ((C[1] >= intersection[1] >= D[1]) or (C[1] <= intersection[1] <= D[1]))):
                    #intersects on the line segments
                    #calculate the distance to the point and store it
                    if(not(overlappingI.__contains__((j,i)))):
                        overlappingI.append((i,j))
                    
            elif(i == 99 and j != 99):
                #First edge
                #print(chromo)
                A = [chromo.genes[i].x, chromo.genes[i].y]
                B = [chromo.genes[0].x, chromo.genes[0].y]
                #Second edge
                C = [chromo.genes[j].x, chromo.genes[j].y]
                D = [chromo.genes[j + 1].x, chromo.genes[j + 1].y]   
                intersection = (line_intersection((A, B), (C, D)))
                #print(str(i) + " " + str(j))
                #check if intersection is on edge line segment
                if(((A[0] >= intersection[0] >= B[0]) or (A[0] <= intersection[0] <= B[0])) and ((A[1] <= intersection[1] <= B[1]) or (A[1] >= intersection[1] >= B[1])) and ((C[0] >= intersection[0] >= D[0]) or (C[0] <= intersection[0] <= D[0])) and ((C[1] >= intersection[1] >= D[1]) or (C[1] <= intersection[1] <= D[1]))):
                    #intersects on the line segments
                    
                    if(not(overlappingI.__contains__((j,i)))):
                        overlappingI.append((i,j))
                        
            elif(j == 99 and i != 99):
                #First edge
                A = [chromo.genes[i].x, chromo.genes[i].y]
                B = [chromo.genes[i + 1].x, chromo.genes[i + 1].y]
                #Second edge
                C = [chromo.genes[j].x, chromo.genes[j].y]
                D = [chromo.genes[0].x, chromo.genes[0].y]  
                intersection = (line_intersection((A, B), (C, D)))
                #check if intersection is on edge line segment
                if(((A[0] >= intersection[0] >= B[0]) or (A[0] <= intersection[0] <= B[0])) and ((A[1] <= intersection[1] <= B[1]) or (A[1] >= intersection[1] >= B[1])) and ((C[0] >= intersection[0] >= D[0]) or (C[0] <= intersection[0] <= D[0])) and ((C[1] >= intersection[1] >= D[1]) or (C[1] <= intersection[1] <= D[1]))):
                    #intersects on the line segments
                    if(not(overlappingI.__contains__((j,i)))):
                        overlappingI.append((i,j))
    return overlappingI
##############################################################
    
x     = []
y     = []
nodes = []
#######
#INPUT
######
    
#data file path
file_path = str(r'C:\Users\burkh\OneDrive\Desktop\AI\Project4\Random100.tsp')
#used to read and parse the tsp file
x, y = read_datafile(file_path)

#create array of nodes
for i in range(len(x)):
    n = node((i + 1), x[i], y[i])
    nodes.append(n)
    
####################
#BEGIN GA PROCESSES
###################
#Population Size
gaPopSize = 200
i = 0
tempArr = []
populationArr = []
prevdist = math.inf

#Create Chromosomes and add to population

###########################
#CREATE INITIAL POPULATION
##########################
for i in range(gaPopSize):
    #Randomly order the nodes to create a chromosome
    populationArr.append(chromosome(i, (np.random.permutation(nodes))))
 
    
numGenertations = 30000
generationNumber = 0
noChange = 0
numNoChangeGen = 1000
worstDist = math.inf

start = time.time()

print("Time" + "\t" + "Best Dist" + "\t" + "Worst Dist" + "\t" + "Deviation")

#Loop through generations
stopping = math.inf
while(((generationNumber < numGenertations) and (noChange < numNoChangeGen)) and stopping != 0):
    #################################
    #TEST FITNESS OF EACH CHROMOSOME
    ################################
    #Measure trip distance
    for i in (range(gaPopSize)):
        #calculate_trip_dist(populationArr[i].listNames())
        trip = populationArr[i].listNames()
        distance = calculate_trip_dist(trip)
        populationArr[i].distance = distance
    
    #############################
    #SELECT FITTEST CHROMOSOMES
    ###########################
    #Percent of chromosomes to mate
    matingPercentage = .5
    #get number of chromosomes to mate
    numMatingChromosomes = getNumMating(gaPopSize, matingPercentage)
    #sort based on fittness -AKA smallest distance is first
    populationArr = sortChromosomes(populationArr)
    
    #Find and select n fittest chromosomes - select the numMatingChromosomes number of chromosomes that are fittest
    matingChromosomes = populationArr[0:numMatingChromosomes]
    
    #Pair mates somewhat randomly
    pairedMates = pairMates(matingChromosomes)
    
    ##################################
    #Graph Current Fittest chromosome
    #################################
    fittest = populationArr[0].listNames()
    x = []
    y = []
    x,y = getXY(fittest)
    #add on return to first node
    x.append(x[0])
    y.append(y[0])
    #check if a new optimum is found
    if(prevdist != populationArr[0].distance):
        #print(populationArr[0].distance)
        currTime = time.time() - start
        bestDist = populationArr[0].distance
        worstDist = populationArr[-1].distance
        stdDeviation = deviation(populationArr)
        #Export data
        print(str(round(currTime,2)) + "\t" + str(round(bestDist,2)) + "\t" + str(round(worstDist,2)) + "\t" + str(round(stdDeviation,2)))
        
        #evaluate stopping Criteria
        stopping = len(findOverlap(populationArr[0]))
        print(stopping)
        
        #Graph
        #graph_coords(x, y, populationArr[0].distance, generationNumber)
        noChange = 0
    else:
        noChange +=1
        
    prevdist = populationArr[0].distance
    #######################
    #CROSSOVER - TWO POINT
    ######################
    
    #Split chromosome at split Point
    tempChromosome1 = []
    tempChromosome2 = []
    i = 0
    for chromo in pairedMates:
        #index of chromosome to be split at for crossover
        splitPoint1, splitPoint2 = generateSplitPoint()
        
        #Get parent genes
        parent1genes = chromo[0].listNames()
        parent2genes = chromo[1].listNames()
        
        #pull first n nodes from parent1 til min split point
        tempChromosome1 = Crossover(parent1genes, parent2genes, splitPoint1, splitPoint2)
        tempChromosome2 = Crossover(parent2genes, parent1genes, splitPoint1, splitPoint2)

        #replace genes in worst performers in initial population
        populationArr[(gaPopSize - i - 1)].genes = tempChromosome1       
        populationArr[(gaPopSize - i - 2)].genes = tempChromosome2     

        i += 2  #increment by two since we are doing two genes per iteration      
            
    ###########
    #MUTATION
    #########
    #1 in mutation chance: chance of mutation occurring on gene
    mutationChance = 1000
    j = 1 #tracks chromosome index - does not include the original best
    for chromo in populationArr[1:]:
        geneList = chromo.listNames()
        i = 0 #tracks gene's index
        
        for gene in geneList:
            #for each gene in chromosome, determine if a mutation occurs
            mutationIndicator = np.random.randint(1,mutationChance)
            
            if(mutationIndicator == 1):
                #Mutation occurred - select gene index randomly to swap with
                swapGeneIndex = np.random.randint(0,len(x)-1)
                #Swap 2 genes
                tempGene = populationArr[j].genes[i]
                populationArr[j].genes[i] = populationArr[j].genes[swapGeneIndex]
                populationArr[j].genes[swapGeneIndex] = tempGene
            i +=1 #track gene's index
        j +=1 #tracks chromosome index
    generationNumber += 1 #Iterate generation number an repeat process until stopping criteria
    