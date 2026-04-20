import numpy as np

# This function creates a moving window of the specified size from the input data frame.
def calculateMovingWindow(dataFrame, window):
    movingWindow = []

    for i in range(len(dataFrame)-window+1):
        windowData = []
        for j in range(window):
            windowData.append(dataFrame[i+j])
        movingWindow.append(windowData)

    # Convert the list of lists to a numpy array for easier manipulation
    return np.array(movingWindow)