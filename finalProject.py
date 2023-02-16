#Importing libraries
import pandas as pd
import numpy as np
from PIL import Image as im
import math
import cv2 as oc


#Function to make an image from Sample file
def makeImageFromFile(filename):
    #Reading the file
    with open(filename, "r") as f:
        temp = f.readlines()
    row = int(temp[0])
    column = int(temp[1])

    #skiprows to skip the first 3 rows that contain row and coloumn size
    temp1 = np.loadtxt(filename, skiprows=3, dtype="float")
    #Shapes the Sample file to the row and coloumn specification according to the forst 2 lines of thhe file
    data = np.reshape(temp1, (row, column))
    image = im.fromarray((data * 255).astype(np.uint8))
    image.save("pic.bmp")


#Function to display the computed image in a pop-up image
def displayColorCoded(array):
    while 1:
        oc.imshow("Image", array)
        
        #waitKey is the function to close the pop-up (using escape key)
        key = oc.waitKey(33)
        if key == 27:
            break


#Function to save a black nd white image from given array
def displayBW(array):
    image = im.fromarray((array * 255).astype(np.uint8))
    image.save("img.png")
    image.show()


#Function to calculate the Correlation Matrix
def correlation(data, row, column):
    R = np.ndarray((row, row))
    
    for i in range(row):  # X Vector
        for j in range(row):  # Y vector
            sumX = 0
            sumY = 0
            sumX2 = 0
            sumY2 = 0
            sumXY = 0
            
            #Calculating the Pearson Coefficent or every row
            for k in range(column):
                sumXY += data[i][k] * data[j][k]
                sumX += data[i][k]
                sumX2 += data[i][k] ** 2
                sumY += data[j][k]
                sumY2 += data[j][k] ** 2
            
            #math.sqrt() to calculate the square root
            R[i][j] = ((column * sumXY) - (sumX * sumY)) / (
                (math.sqrt((column * sumX2) - (sumX**2)))
                * math.sqrt((column * sumY2) - (sumY**2))
            )
            
    return R


#Function to Discritize a file by calculating correlation matrix
def discretization(filename):
    #skiprows to skip the first 3 rows that contain row and coloumn size
    with open(filename, "r") as f:
        temp = f.readlines()
    row = int(temp[0])
    column = int(temp[1])

    temp1 = np.loadtxt(filename, skiprows=3, dtype="float")
    data = np.reshape(temp1, (row, column))

    #Calling the Correlation Matrix function
    R = correlation(data, row, column)

    #Calculating the mean of every coloumn to calculate the Discroitized Matrix
    RMean = R.mean(axis=1)
    boolR = np.ndarray((row, row))

    #Assigning a value of 1 to the values of the correlation 
    #matrix that are larger than the mean of their corresponding coloumn
    for i in range(row):  # ROW
        for j in range(i, row):  # COLUMN
            if R[i][j] < RMean[i]:
                boolR[i][j] = 1
                boolR[j][i] = 1

    #Calling the function to Display Black and White Image
    displayBW(boolR)
    
    #Creating a 2D array with each element containing 
    #an array of length 3 to correspond to values of BGR
    green = np.ndarray((row, row, 3), np.uint8)

    #Assigning a shade of green according to 
    #the values in the Correlation Matrix
    for i in range(row):
        for j in range(row):
            green[i][j][::] = (0, (R[i][j] * 255), 0)

    #Calling the Display Colour Coded Function
    displayColorCoded(green)


#Function to ramdomize/permutate the data and recover it
def permutationRecovery(filename):
    #Reading the file
    with open(filename, "r") as f:
        temp = f.readlines()
    row = int(temp[0])
    column = int(temp[1])

    #skiprows to skip the first 3 rows that contain row and coloumn size
    temp1 = np.loadtxt(filename, skiprows=3, dtype="float")
    data = np.reshape(temp1, (row, column))
    
    #Shuffles the data randomly
    np.random.shuffle(data)

    #Calling the function to calculate the Correlation Matrix and saving it in R
    R = correlation(data, row, column)

    Shuffled = np.array(R)

    #Calculating the mean of the rows 
    RMean = R.mean(axis=1)
    boolR = np.ndarray((row, row))

    #Assigning a value of 1 to the values of the correlation 
    #matrix that are larger than the mean of their corresponding coloumn
    for i in range(row):  # ROW
        for j in range(i, row):  # COLUMN
            if R[i][j] < RMean[i]:
                boolR[i][j] = 1
                boolR[j][i] = 1

    #Creating a 2D array with each element containing 
    #an array of length 3 to correspond to values of BGR
    green = np.ndarray((row, row, 3), np.uint8)

    #Assigning a shade of green according to 
    #the values in the Correlation Matrix
    for i in range(row):
        for j in range(row):
            green[i][j][::] = (0, (R[i][j] * 255), 0)

    #Calculating the signature of the permuted data 
    signature = data.sum(axis=1) * data.mean(axis=1)
    np.reshape(signature, (150, 1))
    
    #Changing the shape of the Signature to turn it into a 150 lengthed array
    new = signature.reshape(150, 1)
    
    #Concatinating the signature array to the permuted data, 
    # every signature will be next to the values from which it was calculated from
    columnAdd = np.concatenate([data, new], axis=1)
    
    #Sorting the permuted data according to the signature value
    sort = columnAdd[columnAdd[:, column].argsort()]
    cdel = np.delete(sort, 4, 1)

    #Calling the function to calculate the Correlation Matrix
    R = correlation(cdel, 150, 4)

    #Calculating the mean of the arrays
    RMean = R.mean(axis=1)
    #Creating a 2D array of zeros
    boolR = np.zeros((row, row))

    #Assigning a value of 1 to the values of the correlation 
    #matrix that are larger than the mean of their corresponding coloumn
    for i in range(row):  # ROW
        for j in range(i, row):  # COLUMN
            if R[i][j] < RMean[i]:
                boolR[i][j] = 1
                boolR[j][i] = 1

    #Creating a 2D array with each element containing 
    #an array of length 3 to correspond to values of BGR
    green = np.ndarray((row, row, 3), np.uint8)

    #Assigning a shade of green according to 
    #the values in the Correlation Matrix
    for i in range(row):
        for j in range(row):
            green[i][j][::] = (0, (R[i][j] * 255), 0)

    # Task3
    print()
    startThreshold = 0.98
    graphArray = np.array(Shuffled)
    z = 1
    while True:
        if np.all((graphArray == 0)):
            break

        print("CLUSTER " + str(z) + ",THRESHOLD: " + str(startThreshold))
        
        #Removing the values that have a value less than the threshold value
        for i in range(row):
            for j in range(i, row):
                if graphArray[i][j] < startThreshold:
                    graphArray[i][j] = 0
                    graphArray[j][i] = 0
                    
                #Making the diagonal 0 to counter every node making a neighbour with itslef
                if i == j:
                    graphArray[i][j] = 0

        #Caluclaing the weight of the nodes, which is the sum of its neighbours
        nodesWeight = graphArray.sum(axis=0)

        #Calculating the highest weight node
        print("Index: ")
        index = 0
        for a in range(row):
            if nodesWeight[a] > index:
                index = a

        print(index)

        print(graphArray)

        #Displayying the neighbours
        print("Neighbors:")
        neighbors = np.array(graphArray[index]).reshape(row)
        print(neighbors)
        
        #Calculating the number of neighbours
        count = 0
        print("Count of neighbors: ")
        for l in range(row):
            if neighbors[l] != 0:
                count += 1
        print(count)

        #Clustering by Making the row and coloumn of the paticular neighbour equal to 0
        for k in range(row):
            if graphArray[index][k] > 0 or graphArray[k][index]:
                for l in range(row):
                    graphArray[l][k] = 0
                    graphArray[k][l] = 0

            graphArray[index][k] = 0
            graphArray[k][index] = 0

        #Keeping the count of clusters
        z += 1
        temp = input("press enter to continue")


# correlation()
permutationRecovery("Sample data-1-IRIS.TXT")
# discretization("Sample data-1-IRIS.TXT")