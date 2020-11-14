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

    plt.plot(x2, y2,'yo-')
    for i in range(len(x2) - 1):
        plt.annotate(i, (x2[i], y2[i]), textcoords="offset points", xytext=(0,5), ha = 'center')
        
    # naming the axes 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    #plt.legend()
    # giving a title to my graph 
    plt.title(("Optimum Distance : " + str(round(min_dist, 2)) + " Gen: " + str(generation)))
      
    # function to show the plot 
    plt.pause(.05)
    plt.show()    

    return
##############################################################
#graph sets of xy coordinates
def graph_coords2(x, y, min_dist, generation):
    #Define graph style
    style.use('dark_background')
    #plt.clf()
    # plotting the points
    plt.plot(x, y,'ro')

    # naming the axes 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    #plt.legend()
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
def calculate_trip_dist(trip, nodes):
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
def getXY(trip, nodes):
    x = []
    y = []
    for gene in trip:
        x.append(nodes[gene -1].x)
        y.append(nodes[gene -1].y)
                      
    return x, y
##############################################################
def getXY2(trip, nodes):
    x = []
    y = []
    for gene in trip:
        x.append(nodes[gene].x)
        y.append(nodes[gene].y)
                      
    return x, y    
##############################################################
def pairMates(matingChromosomes, numMatingChromosomes):
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
def generateSplitPoint(nodes):
    a = np.random.randint(1,len(nodes)-1)
    b = a
    while(a == b):
        b = np.random.randint(1,len(nodes)-1)   
    
    splitPoint1 = min(a, b)
    splitPoint2 = max(a, b) 
    return splitPoint1, splitPoint2
##############################################################
def Crossover(parent1genes, parent2genes, splitPoint1, splitPoint2, nodes):
    tempChromosome = []
    #First section
    for x in range(splitPoint1):
        tempChromosome.append(nodes[parent1genes[x] - 1])
    #Last Section    
    for x in range(splitPoint2, len(nodes)):
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
       return True

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y    
##############################################################
def makeEdgeTable(matingChromosomes, crowd):  
    for generation in matingChromosomes:
        for chromo in generation:
            for i in range(len(chromo.listNames())):
                if(i < len(chromo.listNames()) -1):
                    e1 = min(chromo.genes[i].name - 1, chromo.genes[i+1].name - 1)
                    e2 = max(chromo.genes[i].name - 1, chromo.genes[i+1].name - 1) 
                    #print(str(e1), str(e2))
                else:
                    e1 = min(chromo.genes[i].name - 1, chromo.genes[0].name - 1)
                    e2 = max(chromo.genes[i].name - 1, chromo.genes[0].name - 1)
                    #print(str(e1), str(e2))
                edge = (e1, e2)
                crowd[edge[0]][edge[1]] += 1
    
    return crowd
##############################################################
def getStartingEdge(maxEdge, crowd):
    i = 0
    for column in crowd:
        j = 0
        i +=1
        for row in column:
            if(row > maxEdge[0]):
                #Select max edge in graph. If multiple, first one is selected
                maxEdge[0]    = row
                maxEdge[1] = i
                maxEdge[2] = j
            j += 1
    return maxEdge
##############################################################
def getMaxSuccessor(successors, usedNodes):
    i = 0
    maxEdge = [0, 0] #frequency, i
    for edge in successors:
        if(edge > maxEdge[0] and (not usedNodes.__contains__(edge))):
            maxEdge = [edge, i]
            #print(maxEdge)
        i += 1
      
    return maxEdge
##############################################################
def combineEdges(combinedP90):
    out = []
    used = []
    i=0
    for edge in combinedP90:
        found = False
        j=0
        for edge in combinedP90:
            if(i != j):
                e1 = combinedP90[i]
                e2 = combinedP90[j]
                if(not(used.__contains__(e1) or used.__contains__(e2))):
                    if(e1[0] == e2[0]):
                        #flip the e2 order 
                        e2.reverse()
                        #remove element 1 from e1
                        del e1[0]
                        #join e2 and e1 - store
                        out.append((e2 + e1))
                        used.append(e1)
                        used.append(e2)
                        found = True
                    elif(e1[0] == e2[-1]):
                        del e1[0]
                        out.append((e2 + e1)) 
                        used.append(e1)
                        used.append(e2)                    
                        found = True                    
                    elif(e1[-1] == e2[0]):
                        del e2[0]
                        out.append((e1 + e2))  
                        used.append(e1)
                        used.append(e2)                    
                        found = True                   
                    elif(e1[-1] == e2[-1]):
                        del e1[-1]
                        e1.reverse()
                        out.append((e2 + e1))
                        used.append(e1)
                        used.append(e2)                    
                        found = True    

            j+=1
        if(not found and (not used.__contains__(e1))): #If after every iteration, edge is not found, go to next and append it as it will never find a match
            out.append(e1)                
        i+=1
        
    return out    
##############################################################
def detectOverlap(e1, e2, nodes):
    #Edge1
    n1X = nodes[e1[0]].x
    n1Y = nodes[e1[0]].y
    
    n2X = nodes[e1[1]].x
    n2Y = nodes[e1[1]].y
    
    #Edge2
    n3X = nodes[e2[0]].x
    n3Y = nodes[e2[0]].y
    
    n4X = nodes[e2[1]].x
    n4Y = nodes[e2[1]].y   
    
    #First edge
    A = [n1X, n1Y]
    B = [n2X, n2Y]
    
    #Second edge - the newly formed one
    C = [n3X, n3Y]
    D = [n4X, n4Y]
    
    intersection = list((line_intersection((A, B), (C, D))))
    
    intersection[0] = (round(intersection[0], 5))
    intersection[1] = (round(intersection[1], 5))
    
    if(intersection == True):
        return False
    #check if intersection is on edge line segment
    if(((A[0] >= intersection[0] >= B[0]) or (A[0] <= intersection[0] <= B[0])) and ((A[1] <= intersection[1] <= B[1]) or (A[1] >= intersection[1] >= B[1])) and ((C[0] >= intersection[0] >= D[0]) or (C[0] <= intersection[0] <= D[0])) and ((C[1] >= intersection[1] >= D[1]) or (C[1] <= intersection[1] <= D[1]))):
        #intersects on the line segments
        return True #overlap detected
                
    return False #overlap not detected
##############################################################
def findOverlap(chromo):
    overlappingI = []

    for i in range(len(chromo.listNames())):
        #i 0-99
        for j in range(len(chromo.listNames())):
            #print(i)
            if(j != i and j != i+1 and j!= i-1 and i != (len(chromo.listNames()) -1) and j != (len(chromo.listNames()) -1)):
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
                    
            elif(i == len(chromo.listNames()) -1 and j != len(chromo.listNames()) -1):
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
                        
            elif(j == len(chromo.listNames()) -1 and i != len(chromo.listNames()) -1):
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
def GeneticAlgorithm(gaPopSize, numGenertations, mutationChance, nodes, numNoChangeGen, graph):
        
    ####################
    #BEGIN GA PROCESSES
    ###################
    #Population Size
    i = 0
    populationArr = []
    prevdist = math.inf
    
    #Create Chromosomes and add to population
    
    ###########################
    #CREATE INITIAL POPULATION
    ##########################
    for i in range(gaPopSize):
        #Randomly order the nodes to create a chromosome
        populationArr.append(chromosome(i, (np.random.permutation(nodes))))
     
        
    generationNumber = 0
    noChange = 0
    #numNoChangeGen = 20 #60 10 #300
    #worstDist = math.inf
    
    #print("Time" + "\t" + "Best Dist" + "\t" + "Worst Dist" + "\t" + "Deviation")
    
    #Loop through generations
    #stopping = math.inf
    while(((generationNumber < numGenertations) and (noChange < numNoChangeGen))):
        #################################
        #TEST FITNESS OF EACH CHROMOSOME
        ################################
        #Measure trip distance
        for i in (range(gaPopSize)):
            #calculate_trip_dist(populationArr[i].listNames())
            trip = populationArr[i].listNames()
            distance = calculate_trip_dist(trip, nodes)
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
        pairedMates = pairMates(matingChromosomes, numMatingChromosomes)
        
        ##################################
        #Graph Current Fittest chromosome
        #################################
        fittest = populationArr[0].listNames()
        x = []
        y = []
        x,y = getXY(fittest, nodes)
        #add on return to first node
        x.append(x[0])
        y.append(y[0])
        #check if a new optimum is found
        if(prevdist != populationArr[0].distance):
            #print(populationArr[0].distance)
            bestDist = populationArr[0].distance
            #worstDist = populationArr[-1].distance
            #stdDeviation = deviation(populationArr)
            #Export data
            #print(str(round(currTime,2)) + "\t" + str(round(bestDist,2)) + "\t" + str(round(worstDist,2)) + "\t" + str(round(stdDeviation,2)))
            
            #evaluate stopping Criteria
            #stopping = len(findOverlap(populationArr[0]))
            #print(stopping)
            
            #Graph
            if(graph):
                graph_coords(x, y, populationArr[0].distance, generationNumber)
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
            splitPoint1, splitPoint2 = generateSplitPoint(nodes)
            
            #Get parent genes
            parent1genes = chromo[0].listNames()
            parent2genes = chromo[1].listNames()
            
            #pull first n nodes from parent1 til min split point
            tempChromosome1 = Crossover(parent1genes, parent2genes, splitPoint1, splitPoint2, nodes)
            tempChromosome2 = Crossover(parent2genes, parent1genes, splitPoint1, splitPoint2, nodes)
    
            #replace genes in worst performers in initial population
            populationArr[(gaPopSize - i - 1)].genes = tempChromosome1       
            populationArr[(gaPopSize - i - 2)].genes = tempChromosome2     
    
            i += 2  #increment by two since we are doing two genes per iteration      
                
        ###########
        #MUTATION
        #########
        #1 in mutation chance: chance of mutation occurring on gene
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
    print(bestDist)
    return matingChromosomes, fittest, bestDist
   
##############################################################

xOrig = []
yOrig = []

########################
#TESTING
########################
NumGAPopulations = 2
gaPopSize        = 250

numNoChangeGen   = 50
#data file path
file_path = str(r'C:\Users\burkh\OneDrive\Desktop\AI\Project5\Random97.tsp')
#used to read and parse the tsp file
xOrig, yOrig = read_datafile(file_path)

nodes = []

#create array of nodes
for i in range(len(xOrig)):
    n = node((i + 1), xOrig[i], yOrig[i])
    nodes.append(n)


###########################################################################

#OTHER
#Max Gens
numGenertations = 30000
#1 in mutationChance chance of a mutation on each gene
mutationChance = 1000


matingChromosomes = []
fittest = []

start = time.time()
dists = []
GATimeAvgArr = []

for i in range(NumGAPopulations):
    print("Population: " + str(i + 1))
    GAStart = time.time()
    matingChromosomesRow, fittestRow, bDist = GeneticAlgorithm(gaPopSize, numGenertations, mutationChance, nodes, numNoChangeGen, True)
    dists.append(bDist)
    
    matingChromosomes.append(matingChromosomesRow)
    fittest.append(fittestRow)
    GAEnd = time.time() - GAStart
    GATimeAvgArr.append(GAEnd)
    #print("Total Time: " + str(GAEnd))
    #print("-------------------------")
avgDist = 0
for dist2 in dists:
    avgDist += dist2
avgDist = avgDist / len(dists)
GATimeAvg = 0
for time2 in GATimeAvgArr:
    GATimeAvg += time2
GATimeAvg = GATimeAvg / len(GATimeAvgArr)
#print("AVG Dist: ", avgDist)
#print("AVG Time: ", GATimeAvg)

##############################
#IMPROVE FINAL PATHS WITH WOC
############################# 
#Take fittest Chromosomes - these are now the experts.
#insert and count number of times each edge is used

#Initialze crowd edge matrix
crowd = [] # 0-99 x 0-99 matrix split since it is symmetric #Edges always referenced smallest index first
for i in range(len(nodes)):
    row = []
    for j in range(len(nodes)):
        row.append(0)
    crowd.append(row)

crowd = makeEdgeTable(matingChromosomes, crowd)

#SHOW Crowd Matrix  
#plt.matshow(crowd)
#plt.show() 

for multiplier in range(19):
    multiplier += 1
    mult = multiplier * .5
    #print(mult)
    
    ###############################
    
    #iterate through table and select 90% edge agreement
    i = 0
    Percent90Edges = []
    for column in crowd:
        j = 0
        for row in column:
            if(row > (.1 * mult * NumGAPopulations * len(matingChromosomes[0]))):
                Percent90Edges.append([i,j])
            j+=1
        i+=1
    if(len(Percent90Edges) == 0 or len(Percent90Edges) > len(nodes) -1):
        continue
    
    #Combine segments that share a node 0-99
    improve = math.inf
    while(len(Percent90Edges) < improve and len(Percent90Edges) != 1): #until no improvement is made on an iteration
        improve = len(Percent90Edges)
        Percent90Edges = combineEdges(Percent90Edges)
        
    
    #Find unused nodes
    unusedNodes = [*range(0, len(nodes), 1)] 
    cont = False
    for edge in Percent90Edges:
        if(not isinstance(edge, int)):
            for node in edge:
                if(unusedNodes.__contains__(node)):
                    unusedNodes.remove(node)
                else:
                    cont = True
        else:
            if(unusedNodes.__contains__(edge)):
                unusedNodes.remove(edge)
            else:
                cont = True
    if(cont):
        continue
    #For each unused node, find the closest endpoint and insert
    for node in unusedNodes:
        mindist = math.inf
        minPoint = []
        betterNode = math.inf
        betterDist = math.inf
        i = 0
        for edge in Percent90Edges:
            if(not isinstance(edge, int)):
                end1 = edge[0]
                end2 = edge[-1]
            else:
                end1 = end2 = edge
            #Unused Node coords
            nodeX = nodes[node].x
            nodeY = nodes[node].y
            #Enpoint 1 Node coords
            end1X = nodes[end1].x
            end1Y = nodes[end1].y
            #Enpoint 2 Node coords
            end2X = nodes[end2].x
            end2Y = nodes[end2].y
            #Distance from unused to each endpoint
            dist1 = math.hypot(end1X - nodeX, end1Y - nodeY)
            dist2 = math.hypot(end2X - nodeX, end2Y - nodeY)
    
            if(dist1 > dist2):
                betterNode = end2
                betterDist = dist2
                insertIndex = -1
            elif(dist2 > dist1): #Node 1 is better or equal so just select it
                netterNode = end1
                betterDist = dist1
                insertIndex = 0
                
            if(betterDist < mindist):
                mindist = betterDist
                minPoint = [betterNode, i, insertIndex] #Node, which row of Percent90Edges, which end as an index
    
            i +=1
        Percent90Edges[minPoint[1]].insert(minPoint[2], node)
            
    #Find unused endpoints that are closest together and conect
    
    #Start with first edge in Percent90Edges
    finalTrip = []
    finalTrip = Percent90Edges[0]
    Percent90Edges.remove(finalTrip)
    
    while(len(Percent90Edges) > 0):
        endpoint1 = finalTrip[0]
        endpoint2 = finalTrip[-1]
        #Get XY
        endpoint1X = nodes[endpoint1].x
        endpoint1Y = nodes[endpoint1].y
        
        endpoint2X = nodes[endpoint2].x
        endpoint2Y = nodes[endpoint2].y
        #For each endpoint, find the one that has the closest non-self enpoint node nearby and add that node to the trip
        bestDist = math.inf
        bestEdge = 0
        i = 0
        endIndexNew = math.inf
        endIndexCur = math.inf
        for edge in Percent90Edges:
            
            end1 = edge[0]
            end2 = edge[-1]
            
            end1X = nodes[end1].x
            end1Y = nodes[end1].y
            end2X = nodes[end2].x
            end2Y = nodes[end2].y
            
            #endpoint1
            dist1 = math.hypot(endpoint1X - end1X, endpoint1Y - end1Y) #endpoint1 start - end1 start
            dist2 = math.hypot(endpoint1X - end2X, endpoint1Y - end2Y) #endpoint1 start - end2 end
            #endpoint2
            dist3 = math.hypot(endpoint2X - end1X, endpoint2Y - end1Y) #endpoint1 end   - end1 start
            dist4 = math.hypot(endpoint2X - end2X, endpoint2Y - end2Y) #endpoint1 end   - end2 end    
                
            minDist = min(dist1, dist2, dist3, dist4)
            if(minDist < bestDist):
                if(dist1 == minDist):
                    bestEdge = i #index of edge in Percent90Edges
                    bestDist = minDist
                    endIndexNew = 0
                    endIndexCur = 0
                    
                elif(dist2 == minDist):
                    bestEdge = i #index of edge in Percent90Edges
                    bestDist = minDist
                    endIndexNew = -1    
                    endIndexCur = 0
                elif(dist3 == minDist):
                    bestEdge = i #index of edge in Percent90Edges
                    bestDist = minDist
                    endIndexNew = 0   
                    endIndexCur = -1
                elif(dist4 == minDist):
                    bestEdge = i #index of edge in Percent90Edges
                    bestDist = minDist
                    endIndexNew = -1    
                    endIndexCur = -1
        
            i+=1
        newEdge = Percent90Edges[bestEdge]
        Percent90Edges.remove(newEdge)
        #if(endIndexNew < 0):
            #newEdge.reverse() #Reverse the edge to ad in correct order
        
        if(endIndexCur == -1 and endIndexNew == -1):
            newEdge.reverse()
            finalTrip = finalTrip + newEdge
            #print("1 ran")
        elif(endIndexCur == -1 and endIndexNew == 0):  
            #newEdge.reverse()
            #finalTrip = newEdge + finalTrip
            finalTrip = finalTrip + newEdge 
            #print("2 ran")
        elif(endIndexCur == 0 and endIndexNew == -1):    
            finalTrip = newEdge + finalTrip 
            #print("3 ran")
        elif(endIndexCur == 0 and endIndexNew == 0):    
            newEdge.reverse()
            finalTrip = newEdge + finalTrip
            #print("4 ran")    
        else:
            finalTrip.reverse()
            finalTrip = finalTrip + newEdge  
            print("Error?")
        
    
        
        x2 = []
        y2 = []
        
        x2, y2 = getXY2(finalTrip, nodes)
        
        #Graph
        #graph_coords(x2, y2, 0, 0)            
    
    #Add on return to initial city            
    finalTrip.append(finalTrip[0])            
    x2 = []
    y2 = []
    
    x2, y2 = getXY2(finalTrip, nodes)
    
    #Graph
    #graph_coords(x2, y2, 0, 0)            
    ######################################
    #Detect and 2-swap edges with overlap
    ##################################### 
    reversePerformed = True 
    prevcity1 = math.inf
    prevcity2 = math.inf
    while(reversePerformed):
        overlap = False  
        reversePerformed = False         
        for city1 in range(len(finalTrip) - 1):
            for city2 in range(len(finalTrip) - 1):
                if(city1 != city2 and city1 + 1 != city2 and city1 - 1 != city2): #skip if they are the same edge
                    
                    e1 = [finalTrip[city1], finalTrip[city1 + 1]]
                    e2 = [finalTrip[city2], finalTrip[city2 + 1]]
                    
                    overlap = detectOverlap(e1, e2, nodes)
        
                    if(overlap and city2 != prevcity1 and city1 != prevcity2):

                        #print("City1 = " + str(city1) + " City2 = " + str(city2))
                        swapIndex1 = city1 + 1
                        swapIndex2 = city2
                        
                        revListSegment = list(reversed(finalTrip[min(swapIndex1,swapIndex2):max(swapIndex1,swapIndex2) + 1]))
                        seg1 = finalTrip[0:(min(swapIndex1,swapIndex2))]
                        seg2 = finalTrip[(max(swapIndex1,swapIndex2)) + 1:]
                        
                        finalTrip = seg1 + revListSegment + seg2
                        reversePerformed = True
                        x2 = []
                        y2 = []
                        
                        x2, y2 = getXY2(finalTrip, nodes)
                        
                        #Graph
                        graph_coords(x2, y2, 0, 0)                   
                        break
                    prevcity1 = city1
                    prevcity2 = city2                        
            if(reversePerformed):
                break
    #WC Distance
    currTime = time.time() - start
    calculate_trip_dist(finalTrip, nodes)
    dist = 0
    for j in range(len(finalTrip)-1):
        node1 = nodes[finalTrip[j]]
        node2 = nodes[finalTrip[j+1]]
        
        dist = dist + (math.hypot(node1.x - node2.x, node1.y - node2.y))
    #OUTPUT
    
    
    #print("--------------------------")
    #print("Wisdom of Crowds Solution:")
    #print(str(dist) + " Percent Agreement from Crowd: " + str(mult * 10))
    #print("Total Time: " + str(currTime))
    #print(str(round(avgDist, 2)) + "\t" + str(round(dist, 2)) + "\t" + str(mult * 10) + "\t" + str(round(currTime, 2)) + "\t" + str(round(GATimeAvg, 2)))
    graph_coords(x2, y2, dist, "n/a")
    break #break out since first succesful tends to be best
    ################################
