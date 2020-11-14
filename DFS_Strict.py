#This implementation shows the minimum number of stops that the salesman must take
#It does not take the distance into consideration
#This is a strict implementation of the BFS algorithm

#Import Libraries
from collections import deque
import math
from matplotlib import style    
import matplotlib.pyplot as plt 

#Initialize variables
x = []
y = []
stack = deque()

#Used to store which cities are reachable from each city
cityGraph = { 
          1  : [2, 3, 4],
          2  : [3],
          3  : [4, 5],
          4  : [5, 6, 7],
          5  : [7, 8],
          6  : [8],
          7  : [9, 10],
          8  : [9, 10, 11],
          9  : [11],
          10 : [11],
          11 : []
        }

############################################################################
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
########################################
#graph sets of xy coordinates
def graph_coords(x, y, x2, y2, min_dist):
    #Define graph style
    style.use('dark_background')
    
    # plotting the points
    plt.plot(x2, y2, 'c:')
    plt.plot(x, y,'ro', label="Path")
    for i in range(len(x)):
        plt.annotate((str(i + 1) + ": " + str(cityArr[i].firstVisited) + "/" + str(cityArr[i].lastVisited)), (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha = 'left')
    
    
    # naming the axes 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    #plt.legend()
    # giving a title to my graph 
    plt.title(("DFS Discovery: " + str(min_dist)))
      
    # function to show the plot 
    plt.pause(.05)
    plt.show()    

    return
    
########################################
def DFS(stack, time):
    
    while(stack):
        currCity = stack[-1]
        currCity.firstVisited = time
        cityArrSorted.append(currCity)
        time += 1
        #print(currCity.name)
        for city in currCity.cities:
            if(cityArr[city - 1].firstVisited == math.inf):
                stack.append(cityArr[city - 1])
                time = DFS(stack, time)
            #else ignore
        currCity.lastVisited = time
        cityArrSorted.append(currCity)
        stack.pop()
        time += 1
        return time                
########################################
def getCoordinates(nodeList):
    x = []
    y = []
    for node in nodeList:
        x.append(node.x)
        y.append(node.y)
        
    return x, y
        

########################################
class city():
    def __init__(self, name, x, y, cities):
        self.name = name
        self.x = x
        self.y = y
        self.cities = cities
        self.firstVisited = math.inf
        self.lastVisited = math.inf
        self.previous = 0
        self.distance = math.inf
############################################################################

#######
#INPUT
######
    
#data file path
file_path = str(r'C:\Users\burkh\OneDrive\Desktop\AI\Project2\nodes.tsp')
#used to read and parse the tsp file
x, y = read_datafile(file_path)

#############
#FORMAT DATA
############

#Initalize and store array of cities
cityArr = []
cityArrSorted = []
for i in range(11):
    c = city(i+1, x[i], y[i], cityGraph[i+1])
    cityArr.append(c)

############
#PROCESSING
###########
    
#Initalize Stack with first city
time = 1
stack.append(cityArr[0]) #Starting at node 1 = cityArr[0]

#Run BFS to update city array
DFS(stack, time)

########
#OUTPUT
#######
        
#list path taken
x2, y2 = getCoordinates(cityArrSorted)

#Graph Path/ Unused Points
graph_coords(x, y, x2, y2, cityArr[10].firstVisited)

print("Distance to city 11: " + str(cityArr[10].firstVisited))
#print("Path: " + str((path)).strip('[]'))
