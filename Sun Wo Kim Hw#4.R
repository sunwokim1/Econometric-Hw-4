---
#Group Members: Sun Wo Kim, Leonardo Alcaide, Arifa Begum, Nene Diallo, Maria Camila
title: 'Homework #4'
output: github_document
---
 ### Due 8am EST Friday Sept 27, 2024 
 ### Econ B2000, MA Econometrics
   
load("/Users/jaydenkim/Desktop/Econometrics/Household_Pulse_data_ph4c2.RData")
# Load required libraries
require(tidyverse)
require(class)
require(caret)
require(ggplot2)
library(dplyr)
library(tidyr)
library(plotly)
library(forcats)

# View initial data
View(Household_Pulse_data)

# Create a subset of the data for analysis
Work <- Household_Pulse_data %>%
  select(KINDWORK, LONGCOVID, HADCOVIDRV)
View(Work)

# Recoding 'KINDWORK' to numerical values for 'Type_of_work'
Household_Pulse_data$Type_of_work <- fct_recode(Household_Pulse_data$KINDWORK,
                                                "0" = "NA",
                                                "1" = "work for govt",
                                                "2" = "work for private co",
                                                "3" = "work for nonprofit",
                                                "4" = "self employed",
                                                "5" = "work in family biz")

# Convert 'Type_of_work' into numeric format
Household_Pulse_data$Type_of_work <- as.numeric(levels(Household_Pulse_data$Type_of_work))[Household_Pulse_data$Type_of_work]
summary(Household_Pulse_data$Type_of_work)

# Cross-tabulation of Type_of_work and KINDWORK
xtabs(formula = ~Type_of_work + KINDWORK,  data = Household_Pulse_data)

# Filter data for people who had long COVID
dat_hadcovid <- Household_Pulse_data %>%
  filter(LONGCOVID == "had symptoms 3mo or more Long Covid")
summary(dat_hadcovid)

# Normalize the variables for KNN
norm_varb <- function(X_in) {
  (X_in - min(X_in, na.rm = TRUE)) / (max(X_in, na.rm = TRUE) - min(X_in, na.rm = TRUE))
}

# Preparing the data for KNN classification
data_use_prelim <- data.frame(norm_varb(dat_hadcovid$Type_of_work), norm_varb(dat_hadcovid$Type_of_work))
good_obs_data_use <- complete.cases(data_use_prelim, dat_hadcovid$LONGCOVID)
dat_use <- subset(data_use_prelim, good_obs_data_use)
y_use <- subset(dat_hadcovid$LONGCOVID, good_obs_data_use)

# KNN model split into train and test data
set.seed(02345)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.7)
train_data <- subset(dat_use, select1)
test_data <- subset(dat_use, (!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]

# KNN classification for different k values
k_values <- seq(1, 20, by = 2)
accuracy_results <- data.frame(k = k_values, accuracy = rep(0, length(k_values)))

for (i in seq_along(k_values)) {
  pred_knn <- knn(train_data, test_data, cl = cl_data, k = k_values[i])
  accuracy_results$accuracy[i] <- mean(pred_knn == true_data)
}

# 1. KNN Classification Accuracy Plot
knn_plot <- ggplot(accuracy_results, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point(size = 3, color = "blue") +
  labs(x = "k (Number of Neighbors)", y = "Accuracy", title = "KNN Classification Accuracy for Different k Values") +
  theme_minimal()

# 2. Long COVID Plot
longcovid_plot <- ggplot(Household_Pulse_data, aes(x = as.factor(Type_of_work), fill = as.factor(LONGCOVID))) +
  geom_bar(position = "dodge") +
  labs(x = "Type of Work", y = "Count", fill = "Long COVID Status",
       title = "Relationship Between Type of Work and Long COVID") +
  scale_fill_manual(values = c("orange", "green"), labels = c("No", "Yes")) +
  theme_minimal()

# 3. Type of Work Plot
type_of_work_plot <- ggplot(Household_Pulse_data, aes(x = as.factor(Type_of_work))) +
  geom_bar(fill = "skyblue") +
  labs(x = "Type of Work", y = "Count", title = "Distribution of Type of Work") +
  theme_minimal()

# ---------------------------- Show the Plots ---------------------------- #

# Print all the plots
print(knn_plot)
print(longcovid_plot)
print(type_of_work_plot)


#In this KNN lab, we recoded the GENID_DESCRIBE column into numeric values and normalized the variables to prepare the data for classification.
#After filtering for individuals who tested positive for COVID-19, we split the data into training and testing sets.
#As the value of k increased, accuracy initially improved but then stabilized, indicating diminishing returns for larger k values.
#We also discovered interesting trends, such as variations in long COVID rates across different gender identities. 
#Overall, the results highlighted the importance of selecting an optimal k and suggested potential links between gender identity and long COVID.

