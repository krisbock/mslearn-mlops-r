library("carrier")

accidents <- readRDS("experimentation/data/accidents.Rd")
summary(accidents)

# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- sample(1:nrow(accidents), 0.75 * nrow(accidents))
train <- accidents[sampled, ]
test <- accidents[-sampled, ]


if (!dir.exists("production/data")) dir.create("production/data", recursive = TRUE)

write.csv(train, file.path("production/data/train.csv"))
write.csv(test, file.path("production/data/test.csv"))

model <- glm(dead ~ dvcat + seatbelt + frontal + sex + ageOFocc + yearVeh + airbag  + occRole, family=binomial, data=train)
summary(model)

predictor <- crate(~ factor(ifelse(stats::predict(!!model, .x)>0.1, "dead","alive")))
predictions <- predictor(test)
accuracy <- mean(predictions == accidents$dead)
accuracy
