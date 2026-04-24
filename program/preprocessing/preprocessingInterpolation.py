import pandas as pd


def preprocessingInterpolation(dataFrame):

    print("[INFO] Starting interpolation...")
    dataFrame = dataFrame.asfreq('D')
    dataFrame.interpolate(method='linear', inplace=True)

    return dataFrame