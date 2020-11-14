#Import libraries
from itertools import permutations
from matplotlib import style    
import math
import matplotlib.pyplot as plt 

####################################################################################
#Declare empty arrays to store x,y coordinates
least_distance = math.inf
bestx = []
besty = []
best_path = []
x = []
y = []

####################################################################################
#function definitions

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

#######################################################
#graph sets of xy coordinates
def graph_coords(x, y, min_dist):
    #Define graph style
    style.use('dark_background')
    
    # plotting the points
    plt.plot(x, y,'ro-')
    for i in range(len(x) - 1):
        plt.annotate(i + 1, (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha = 'center')
        
    # naming the axes 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
      
    # giving a title to my graph 
    plt.title(("Optimum Path Length: " + str(min_dist)))
      
    # function to show the plot 
    plt.pause(.05)
    plt.show()    

    return

#######################################################
#Convert the indices to actual point numbers as in the tsp file
def index2point(path):
    out = []
    for element in path:
        out.append(element+1)        
    return out

#######################################################
#Calculate distance for the trip
def calculate_trip_dist(tripx, tripy):
    dist = 0
    for i in range(len(tripx) - 1):
        dist = dist + (math.hypot(tripx[i] - tripx[(i+1)], tripy[i] - tripy[i+1]))
    return dist

####################################################################################

############
#INPUT 
###########

#data file path
file_path = str(r'C:\Users\burkh\OneDrive\Desktop\AI\Project1\datasets\Random4.tsp')
#used to read and parse the tsp file
x, y = read_datafile(file_path)


############
#PROCESSING
###########

#Create all possible paths
perm = permutations(range(len(x)))

#iterate through permutations
for p in perm: 
    tripx = []
    tripy = []
    
    #record trip using permutations of the range of 0 -> length(x) as pointers
    for element in p:
        tripx.append(x[element])
        tripy.append(y[element])
        
    #take trip back to starting node  
    tripx.append(x[p[0]])
    tripy.append(y[p[0]])
    
    #Calculate distance for the trip
    temp_dist = 0
    temp_dist = calculate_trip_dist(tripx, tripy)
    
    #store lowest distance and trip
    if(temp_dist < least_distance):
        least_distance = temp_dist
        #Store current optimums
        bestx = tripx
        besty = tripy
        best_path = p
        
############
#OUTPUT
###########
        
#Create graph of the optimum path       
graph_coords(bestx, besty, least_distance)

#Send output for report
print(least_distance)
print(index2point(best_path)) #Write best path using point numbers