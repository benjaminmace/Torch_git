import os

for subdir, dirs, files in os.walk('dataset'):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        print(filepath)