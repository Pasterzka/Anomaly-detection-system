# This module contains functions for splitting the data into training, validation, and test sets, as well as normalizing the data.
def splitAndNormalizeData(dataFrame):
    
    n = len(dataFrame)
    trainEnd = int(n * 0.7)
    valEnd = int(n * 0.85)

    dataFrameTrain = dataFrame[:trainEnd]
    dataFrameVal = dataFrame[trainEnd:valEnd]
    dataFrameTest = dataFrame[valEnd:]

    for columnName in dataFrame.columns:
        trainMin = dataFrameTrain[columnName].min()
        trainMax = dataFrameTrain[columnName].max()

        dataFrameTrain = normalizeData(dataFrameTrain, columnName, trainMin, trainMax)
        dataFrameTest = normalizeData(dataFrameTest, columnName, trainMin, trainMax)
        dataFrameVal = normalizeData(dataFrameVal, columnName, trainMin, trainMax)

    return dataFrameTrain, dataFrameVal, dataFrameTest

# This function normalizes the data using min-max normalization.
def normalizeData(dataFrame, columnName, minValue, maxValue):
    df = dataFrame[columnName].copy()


    # Avoid division by zero
    denominator = maxValue - minValue
    if denominator == 0:
        denominator = 1e-10

    df = (df - minValue) / (maxValue - minValue)

    dataFrame[columnName] = df

    return dataFrame
