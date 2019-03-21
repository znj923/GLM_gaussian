from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession\
    .builder\
    .appName("GeneralizedLinearRegressionExample")\
    .getOrCreate()

# Load  data
data = spark.read.csv("file:///D:/Spark/spark-2.3.3-bin-hadoop2.7/data/Linear_regression_house-master/boston.csv", header=True, inferSchema=True)
data.show(10)


# 合并特征
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                                             'ptratio', 'black', 'lstat'], outputCol='features')
v_data = vectorAssembler.transform(data)
v_data.show(10)

# 划分训练集，集测试集
vdata = v_data.select(['features', 'medv'])
vdata.show(10)
splits = vdata.randomSplit([0.7, 0.3])
train_data = splits[0]
test_data = splits[1]

# 训练
glr = GeneralizedLinearRegression(family="gaussian", link="identity", labelCol='medv',featuresCol='features', maxIter=1000, regParam=0.3)
# Fit the model
GlModel = glr.fit(train_data)

# Print the coefficients and intercept for generalized linear regression model
print("Coefficients: " + str(GlModel.coefficients))
print("Intercept: " + str(GlModel.intercept))

# Summarize the model over the training set and print out some metrics
summary = GlModel.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()

# 预测
predictions = GlModel.transform(test_data)
predictions.select('features', 'medv', 'prediction').show()

# 评估
lr_evaluator = RegressionEvaluator(metricName="r2", predictionCol='prediction', labelCol='medv')
r2 = lr_evaluator.evaluate(predictions)
test_evaluation = GlModel.evaluate(test_data)
print("r2: %f" % r2)

spark.stop()
