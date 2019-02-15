# This script is aim to preprocess the data from weather

library(imputeTS)
library(missForest)

args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0) {
  stop("Need to specify input file and output file.\n", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = "out.csv"
}

RAW_WEATHER_DATA_PATH <- args[1]
CLEANED_DATA_PATH <- args[2]
COL_NAMES <- c("DateTime", "Condition", "Temperature", "Wind", "Humidity", "Visibility")

data <- read.csv(file=RAW_WEATHER_DATA_PATH, header=FALSE, sep=",", col.names = COL_NAMES, na.strings=c("","NA"))

#data$Condition <- as.factor(data$Condition)
#data$Visibility <- as.ts(data$Visibility)
#data$Temperature <- as.ts(data$Temperature)
#data$Wind <- as.ts(data$Wind)
#data$Humidity <- as.ts(data$Humidity)

summary(data)

data$Visibility = na.interpolation(data$Visibility, option = "linear")
data$Wind = na.interpolation(data$Wind, option = "linear")
mf = missForest(data[-1], verbose = TRUE)$ximp

summary(mf)

#library(DMwR)
#data.knn <- knnImputation(data[-1], k = 3, scale = TRUE)
data$Condition <- mf$Condition

#library(FactoMineR)
#data$Condition <- as.factor(data$Condition)
#PCA(data[-1], quali.sup = 1)

write.csv(data, file = CLEANED_DATA_PATH, row.names = FALSE, quote=FALSE)

