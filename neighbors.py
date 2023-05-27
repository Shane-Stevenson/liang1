import numpy as np
import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

las = laspy.read('initData.las')
print(las) 

visited = []
queue = [] 
finlist = []

xoffset = las.header.offsets[0]
xscale = las.header.scales[0]

yoffset = las.header.offsets[1]
yscale = las.header.scales[1]

zoffset = las.header.offsets[2]
zscale = las.header.scales[2]


count = 0

for n in las: 

    # 500 trees? break! 
    if (count > 500): break

    # medium + high vegetation 
    if n.classification == 4 or n.classification == 5: 
        
        # counts every time there's a new cluster of green 
        count += 1 
        print(count)

        # list of This Bunch Of Green Stuff 
        templist = []

        # throw 'starting value' into the queue/visited/&c 
        visited.append(n)
        queue.append(n)
        templist.append(n)

        while queue: 

            # basic bfs 
            m = queue.pop(0)
            
            #look for points distance away
            distance = .5
            
            #Find neighbors 
            

            # if neighbor not in visited:
            #     if lasneigh.classification == 4 or lasneigh.classification == 5:  
            #         visited.append(neighbor)
            #         queue.append(neighbor)
            #         templist.append(neighbor)
            #         print(neighbor)
            #         print(templist)

        finlist.append(templist)

print("Rendering stage.")

#for i in finlist: 
    #for n in i: 
        #print(n)

# turn it into a numpy array for 3d slicing(?) purposes 
#nplist = np.array(finlist)
#test = finlist[0]

#print(finlist[0])

#print(x_data)
#print(y_data)
#print(z_data)

#fig = plt.figure(figsize=(5, 5))
#ax = fig.add_subplot(111, projection="3d")

#ax.scatter(x_data, y_data, z_data)
#ax.set_axis_off()
#plt.show()