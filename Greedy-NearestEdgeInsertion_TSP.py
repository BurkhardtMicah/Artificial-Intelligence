#CLOSEST EDGE INSERTION HEURISTIC
import math
from matplotlib import style    
import matplotlib.pyplot as plt 
from itertools import permutations

##############################################################
class node():
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.used = False

########################################
#graph sets of xy coordinates
def graph_coords(x, y, x2, y2, min_dist):
    #Define graph style
    style.use('dark_background')
    plt.clf()
    # plotting the points
    plt.plot(x, y,'ro', label="Non-Visited Vertices")
    plt.plot(x2, y2,'yo-', label="Optimum Path")
    for i in range(len(x2) - 1):
        plt.annotate(i + 1, (x2[i], y2[i]), textcoords="offset points", xytext=(0,5), ha = 'center')
        
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
#############################################################
#Calculate distance for the trip
def calculate_trip_dist(trip):
    dist = 0
    for i in range(len(trip) - 1):
        dist = dist + (math.hypot(trip[i].x - trip[(i+1)].x, trip[i].y - trip[i+1].y))
    return dist
##############################################################
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
#############################################################

x           = []
y           = []
x2          = []
y2          = []
nodes       = []
currentPath = []

#######
#INPUT
######
    
#data file path
file_path = str(r'C:\Users\burkh\OneDrive\Desktop\AI\Project3\Random30.tsp')
#used to read and parse the tsp file
x, y = read_datafile(file_path)

############
#PROCESSING
###########

#initialize array of nodes
for i in range(len(x)):
    n = node((i + 1), x[i], y[i])
    nodes.append(n)
#Put furthest three nodes first
perm = permutations(range(len(x)), 3)
most = math.inf
bestPerm = None

for p in perm:
    dist = 0
    for i in range(len(p)):
        if(i < 2):
            dist = dist + (math.hypot(nodes[p[i]].x - nodes[p[i + 1]].x, nodes[p[i]].y - nodes[p[i + 1]].y))
        else:
            dist = dist + (math.hypot(nodes[p[i]].x - nodes[p[0]].x, nodes[p[i]].y - nodes[p[0]].y))  
        #print(i)
    if(most > dist):
        most     = dist
        bestPerm = p

    
#SELECT STARTING POINTS - take first two points so that starting path is pseudo-random
for node in bestPerm:
    currentPath.append(nodes[node])
    nodes[node].used = True
    x2.append(nodes[node].x)
    y2.append(nodes[node].y)
    
currentPath.append(nodes[bestPerm[0]]) #return to first node to close circuit
x2.append(nodes[bestPerm[0]].x)
y2.append(nodes[bestPerm[0]].y)

#Iterate through edges
edge2Test = 0
iteration = 1
changed   = False
neighborhood = 1 #limits how far we can search from an edge

while(len(currentPath) < (len(x) + 1)):
    #Reset variables
    edge         = 0
    edge2Test    = 0
    nodeNotSet   = 1
    
    if(iteration != 1 and changed == False):
        #If no changes - increase search radius by 5
        neighborhood = neighborhood + 1
        #print("Neighborhood increased to " + str(neighborhood))
    else:
        neighborhood = 1
    changed = False
    
    for edge in range(len(currentPath) - 1):
        edge2Test = ((edge * 2) + nodeNotSet)
        
        #Skip the newly inserted edge and select the 2 nodes that make the edge
        n1 = currentPath[edge2Test - 1]
        n2 = currentPath[edge2Test]
            
        #Get slope of edge
        slope = ((n1.y - n2.y) / (n1.x - n2.x)).as_integer_ratio()
        
        #Get opposite reciprocal slope
        if(slope[0] /slope[1] < 0):
            perp_slope = [abs(slope[0]), abs(slope[1])]
        else:
            perp_slope = [-slope[0], slope[1]]
        
        #iterate through unused nodes
        
        min_distance = math.inf
        best_node    = None
        for node in nodes:
            if(node.used == False):
                
                b = node.y - (perp_slope[0]/perp_slope[1]) * node.x 
                
                A = [n1.x, n1.y]
                B = [n2.x, n2.y]
                
                C = [node.x, node.y]
                D = [0, b]
                
                #Find Y intercept for each unused node
                intersection = (line_intersection((A, B), (C, D)))
                
                #check if intersection is on edge line segment
                if(((n1.x <= intersection[0] <= n2.x) or (n1.x >= intersection[0] >= n2.x)) and ((n1.y <= intersection[1] <= n2.y) or (n1.y >= intersection[1] >= n2.y))):
                    #intersects on the line segment
    
                    #calculate the distance to the point and store it
                    distance = (math.hypot(node.x - intersection[0], node.y - intersection[1]))
                else:
                    #Calculate distance to each end point and store the smaller
                    dist1 = (math.hypot(n1.x - node.x, n1.y - node.y))
                    dist2 = (math.hypot(n2.x - node.x, n2.y - node.y))
                    
                    if(dist1 < dist2):
                        distance = dist1
                    elif(dist1 > dist2):
                        distance = dist2
                    else:
                        #They are equal - choose either distance
                        distance = dist1
                        
                #compare distance and select closest point to segment to add between points
                if((distance < min_distance) and (distance < neighborhood)): #and distance < 40
                    min_distance = distance
                    best_node = node.name
                    
        #insertIndex = insertIndex + 1
        #insertedEdges = insertedEdges + 1
        if(best_node != None):
            currentPath.insert(edge2Test, nodes[best_node - 1])
            x2.insert(edge2Test, nodes[best_node - 1].x)
            y2.insert(edge2Test, nodes[best_node - 1].y)
            nodes[best_node - 1].used = True
            changed = True
        else:
            nodeNotSet = nodeNotSet - 1
    
########
#OUTPUT 
#######
        if(changed == True):
            total_dist = calculate_trip_dist(currentPath)
            graph_coords(x, y, x2, y2, total_dist) 
            
    iteration = iteration + 1