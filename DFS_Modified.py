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
    plt.plot(x, y,'ro', label="Non-Visited Vertices")
    plt.plot(x2, y2,'yo-', label="Optimum Path")
    for i in range(len(x2)):
        plt.annotate((len(x2) - (i + 1)), (x2[i], y2[i]), textcoords="offset points", xytext=(0,5), ha = 'center')
        
    # naming the axes 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.legend()
    # giving a title to my graph 
    plt.title(("Optimum Distance : " + str(min_dist)))
      
    # function to show the plot 
    plt.pause(.05)
    plt.show()    

    return
    
########################################
def DFS_modified(stack):
    
    while(stack):
        currCity = stack[-1]
        #currCity.firstVisited = time
        cityArrSorted.append(currCity)
        #time += 1
        #print(currCity.name)
        for city in currCity.cities: #Grab left most element from stack
            
            
            
            
            #Check if the new distance to the city is better than the old best distance
            distance = calculate_dist(currCity, cityArr[city - 1])
            #print(distance)
            #if a better distance is found to the city, add it to the array to update paths to next cities if better
            #store best distance and predecessor for backtracking
            if(cityArr[city - 1].distance > distance): 
                #print(cityArr[city - 1].name)
                stack.append(cityArr[city - 1])
                cityArr[city - 1].distance = distance
                cityArr[city - 1].previous = currCity.name
                DFS_modified(stack)
            #else ignore
        #currCity.lastVisited = time
        cityArrSorted.append(currCity)
        if(stack):
            stack.pop()
        #time += 1
        #return time                

########################################
def backtrack(cityArr):
    i = 11
    out = []
    out.append(i)
    
    xcoords = []
    xcoords.append(cityArr[i - 1].x)
    ycoords = []
    ycoords.append(cityArr[i - 1].y)
    
    while(cityArr[i - 1].previous != 0): 
        i = cityArr[i - 1].previous
        out.append(i)
        xcoords.append(cityArr[i - 1].x)
        ycoords.append(cityArr[i - 1].y)
    
    return xcoords, ycoords, out    

########################################
def calculate_dist(city1, city2):
    dist = city1.distance
    dist = dist + (math.hypot(city1.x - city2.x, city1.y - city2.y))
    return dist           

########################################
class city():
    def __init__(self, name, x, y, cities):
        self.name = name
        self.x = x
        self.y = y
        self.cities = cities
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
cityArr[0].distance = 0 #Set distance to 0 - measuring distance from city 1
stack.append(cityArr[0]) #Starting at node 1 = cityArr[0]

#Run BFS to update city array
DFS_modified(stack)

########
#OUTPUT
#######
        
#list path taken
sequence = []
x1, y1, sequence = backtrack(cityArr)

#Graph Path/ Unused Points
graph_coords(x, y, x1, y1, cityArr[10].distance)

print("Distance to city 11: " + str(cityArr[10].distance))
#print("Path: " + str((path)).strip('[]'))
