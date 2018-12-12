from finetune import Classifier
import pandas
from sklearn.model_selection import train_test_split
import time
import re, string
import numpy as np
print("Import of packages done")

filePath = "/root/combined_trainingdata_20181108.tsv"
data = pandas.read_csv(filePath,sep='\t')
print(data.shape)
print("Loaded the data")

print(data.columns[data.isna().any()].tolist())
print(data[data.COMPLAINT_1.isna()].shape)
print(data[data.COMPLAINT_2.isna()].shape)
print(data[(data.COMPLAINT_1.isna()) & (data.COMPLAINT_2.isna())].shape)

dataFiltered = data.dropna(subset = ["COMPLAINT_1"])
print(dataFiltered[(dataFiltered.COMPLAINT_1.isna())].shape)
print(dataFiltered[(dataFiltered.COMPLAINT_1.isna()) & (dataFiltered.COMPLAINT_2.isna())].shape)

translator = str.maketrans('', '', string.punctuation) # To remove punctuation

def preProcess(complaintStart):
    complaint = complaintStart[:512] # cut to 512 characters max
    complaint = re.sub("\d","N", complaint) # remove numbers
    complaint = complaint.lower().translate(translator) # lower case and remove the punctuation
    complaint = complaint.replace("\n"," ").strip() # remove starting and trailing white spaces
    if re.search('[a-zA-Z]', complaint) is None:# if there are no letters in the complaint, return empty, will be removed in later processing
        return ""
    return complaint

def getComplaint(row):
    complaint2 = row.get("COMPLAINT_2")
    if not pandas.isnull(complaint2):
        if "[INSPECTION LOG #:" in complaint2: # Remove inspection log section from C2
            complaintStrippedList = complaint2.split("]")[1:]
            complaintFinal = "]".join(complaintStrippedList)
        else:
            complaintFinal = complaint2
        if row.get("CITY")=="US_CHICAGO": # if Chicago, concatenate the two
            complaintFinal = row.get("COMPLAINT_1") + " "+ complaintFinal
        complaintProcessed = preProcess(complaintFinal)
        if complaintProcessed == "" or re.search('[a-zA-Z]', complaintProcessed) is None: # if nothing or no letters
            return preProcess(row.get("COMPLAINT_1"))
        return complaintProcessed
    complaintProcessed = preProcess(row.get("COMPLAINT_1"))
    return complaintProcessed

results = dataFiltered.apply(lambda row: getComplaint (row),axis=1)
print(results[results.isna()].shape)

dataFiltered["complaint"] = results
print(dataFiltered[dataFiltered.complaint.isna()].shape)
print("Preprocessed and Created a single complaint column")

dataFiltered["CATEGORY_SUB"] = dataFiltered["CATEGORY_SUB"].str.strip()
print("Removed white spaces from CATEGORY_SUB")

mask = (dataFiltered["complaint"].str.len() >=10)
dataFiltered = dataFiltered.loc[mask]
print(dataFiltered.shape)
print("Removed the shortest complaints")

aggregation = {"complaint":"count"}
aggregatedByLabel = dataFiltered.groupby("CATEGORY_SUB").agg(aggregation)

goodLabels = aggregatedByLabel[aggregatedByLabel["complaint"]>100]
goodLabelsList = goodLabels.index.tolist()
print(dataFiltered.shape)
dataGoodLabels = dataFiltered[dataFiltered["CATEGORY_SUB"].isin(goodLabelsList)]
print(dataGoodLabels.shape)
print("Removed labels with too few samples")
del dataFiltered

labelsMap = dataGoodLabels[["CATEGORY_MAIN", "CATEGORY_SUB"]].drop_duplicates()
labelsMap = labelsMap.set_index("CATEGORY_SUB").to_dict()["CATEGORY_MAIN"]
print("Calculated Mapping from MAIN to SUB category")

trainingData = dataGoodLabels[["complaint", "CATEGORY_SUB"]]
del dataGoodLabels
print(type(trainingData))
print(trainingData.shape)
print("Prepared training data")

_, sampleX, _, sampleY = train_test_split(trainingData.complaint, trainingData.CATEGORY_SUB, test_size=0.1, random_state=42, stratify=trainingData.CATEGORY_SUB)
del trainingData
print(sampleY.shape)
print("Prepared a stratified sample. TODO check it is truly so")

trainX, testX, trainY, testY = train_test_split(sampleX, sampleY, test_size=0.2, random_state=42, stratify=sampleY)
print(trainX.shape)
print("Split into train and test")
print("starting gridsearch")
res = Classifier.finetune_grid_search_cv(trainX.tolist(), trainY.tolist(), n_splits=2,
	eval_fn=lambda y1, y2: np.mean(np.asarray(y1) == np.asarray(y2)),test_size=0.1,return_all=True)

print("Results:")
print(res)
exit(0)

print("Starting training")
print(trainX.shape)
start = time.time()
model = Classifier(max_length=512, val_interval=3000, verbose = True)               # Load base model
model.fit(trainX.tolist(), trainY.tolist())          # Finetune base model on custom data
duration = time.time()-start
print("Training Done")
print("It took :"+str(duration)+ " seconds")

model.save("combined_model_20181018")                   # Serialize the model to disk
print("Model Saved")

# model = Classifier.load("../models/combined_model_20181018")
print(testX.shape)
print(model)
start = time.time()
predictions = model.predict(testX.tolist())
duration = time.time() - start
print("Predictions done")
print("It took :"+str(duration)+ " seconds")

mainPredictions = []
for pred in predictions:
    mainPredictions.append(labelsMap[pred])

mainTestY = []
for testLabel in testY.tolist():
    mainTestY.append(labelsMap[testLabel])
    
correctMain = 0
countMain = 0
for i, complaint in enumerate(testX.tolist()):
    correctMain += int(mainPredictions[i] == mainTestY[i])
    countMain +=1
print(correctMain)
print(countMain)
print("Accuracy on Main: "+str(correctMain*1.0/countMain))

correct = 0
count = 0
testYList = testY.tolist()
for i, complaint in enumerate(testX.tolist()):
    correct += int(predictions[i] == testYList[i])
    count +=1
print(correct)
print(count)
print("Accuracy on Sub: "+str(correct*1.0/count))
