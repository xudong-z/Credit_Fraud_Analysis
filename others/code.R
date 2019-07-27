library(data.table)
library(dplyr)
library(ggplot2)


#read
data = fread('creditcard.csv')

#exploration
summary(data)
str(data)
table(data$Class)
sum(is.na(data))
boxplot(data$Amount[data$Class == "1"]) #check fraud amount boxplot

# density check
plot(density(data$V1))
boxplot(data$V1~data$Class) # V1跟Y的关系， boxplot
ggplot(data, aes(x = V1)) + # V1跟Y的关系， density
    geom_density(aes(group = Class, fill = Class, alpha = .3))

ggplot(data[data$Class==1,], aes(x = Amount)) + # Amount跟Y的关系， density
    geom_density(aes(group = Class, fill = Class, alpha = .3))  
## fraud amount is usually no more than USD2000
## most amounts of fraud event<1000

ggplot(data[data$Class==0,], aes(x = Amount)) + # Amount跟Y的关系， density
    geom_density(aes(group = Class, fill = Class, alpha = .3))

# 重点检查amount
boxplot(data$Amount~data$Class) # amount跟Y的关系， boxplot
boxplot(data[data$Class == "1"]$Amount~
            data[data$Class == "1"]$Class) # amount跟Y的关系， boxplot
# fraud amount is usually no more than USD2000


#predictive modeling
