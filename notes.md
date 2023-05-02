# create aws profile
aws configure --profile cs643

aws s3api create-bucket \
--bucket p2-cs643-bucket \
--region us-east-1


s3://p2-cs643-bucket/validation.csv
s3://p2-cs643-bucket/training.csv

# start my spark cluster

aws ec2 describe-subnets \
--filters "Name=availabilityZone,Values=us-east-1a"

`subnetId: subnet-025c1de09afa7571b`

aws emr create-cluster \
--name “CS643 Spark cluster” \
--release-label emr-6.0.0 \
--applications Name=Spark \
--steps Type=Spark,Name=”CS643 Spark cluster”,ActionOnFailure=CONTINUE,Args=[ — class,com.njit.edu.cs643.v2.WinePredictionTraining,s3://p2-cs643-bucket/wineprediction-1.0-SNAPSHOT.jar,s3://p2-cs643-bucket/hubert/training.csv,s3://p2-cs643-bucket/hubert/output] \
--use-default-roles \
--ec2-attributes SubnetId=subnet-025c1de09afa7571b --instance-type m4.large --instance-count 3 \
--auto-terminate


aws emr create-cluster --name "Training Cluster" \
--applications Name=Spark \
--use-default-roles \
--release-label emr-6.0.0 \
--ec2-attributes KeyName=spark \
--log-uri s3://myBucket/myLog \
--ec2-attributes SubnetId=subnet-025c1de09afa7571b \
--instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge InstanceGroupType=CORE,InstanceCount=2,InstanceType=m5.xlarge \
--steps '[{"Args":["spark-submit","--deploy-mode","cluster","--class","com.njit.edu.cs643.v2.WinePredictionTraining","s3://p2-cs643-bucket/wineprediction-1.0-4.jar","s3://p2-cs643-bucket/hubert/training.csv","s3://p2-cs643-bucket/hubert/output","5"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Training"}, {"Args":["spark-submit","--deploy-mode","cluster","--class","com.njit.edu.cs643.v2.WinePredictionTraining","s3://p2-cs643-bucket/wineprediction-1.0-4.jar","s3://p2-cs643-bucket/hubert/training.csv","s3://p2-cs643-bucket/hubert/output","5"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Training"}]'
--auto-terminate



aws emr create-cluster \
--release-label emr-6.0.0 \
--applications Name=Spark \
--ec2-attributes KeyName=spark \
--instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m4.large InstanceGroupType=CORE,InstanceCount=2,InstanceType=m4.large \
--auto-terminate


aws emr create-cluster --name "CS643 Spark cluster" --release-label emr-6.0.0 --applications Name=Spark \
--ec2-attributes KeyName=spark --instance-type m4.large --instance-count 3 --use-default-roles

com.njit.edu.cs643.v2.WinePredictionTraining
com.njit.edu.cs643.v2.WinePredictionValidation
s3://p2-cs643-bucket/wineprediction-1.0-SNAPSHOT.jar
s3://p2-cs643-bucket/hubert/training.csv
s3://p2-cs643-bucket/hubert/output

# Folder 

base folder:
s3://p2-cs643-bucket/hubert/

Input folder: s3://p2-cs643-bucket/hubert/training.csv
outputFolder: s3://p2-cs643-bucket/hubert/output


50.19.38.138

spark-submit \
--class com.njit.edu.cs643.v2.WinePredictionTraining \
--master "spark:50.19.38.138" \
--deploy-mode cluster \
s3://p2-cs643-bucket/wineprediction-1.0-SNAPSHOT.jar \
s3://p2-cs643-bucket/hubert/training.csv \
s3://p2-cs643-bucket/hubert/output



# TODO

 - A link to your code in GitHub. The code includes the code for parallel model training and the code  for prediction application. ( done)
 - A link to your container in Docker Hub.
 - describe step-by-step how to setup the cloud environment
 - describe how to run the model training
 - describe how to run the application prediction without docker
 - describe how to run the application prediction with docker


This file must also describe step-by-step how to set-up the cloud environment and run the model training
and the application prediction. For the application prediction, you should provide instructions on how to
run it with and without Docker





### to train the model. working fine

aws emr create-cluster --name "Training Cluster" \
--applications Name=Spark \
--use-default-roles \
--release-label emr-6.0.0 \
--ec2-attributes KeyName=spark \
--log-uri s3://p2-cs643-bucket/logs/ \
--ec2-attributes SubnetId=subnet-025c1de09afa7571b \
--instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge InstanceGroupType=CORE,InstanceCount=2,InstanceType=m5.xlarge \
--steps '[{"Args":["spark-submit","--deploy-mode","cluster","--class","com.njit.edu.cs643.v2.WinePredictionTraining","s3://p2-cs643-bucket/wineprediction-1.0-4.jar","s3://p2-cs643-bucket/hubert/training.csv","s3://p2-cs643-bucket/hubert/output","5"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Spark application"}]'


aws emr create-cluster --name "Hubert Cluster" \
--applications Name=Spark \
--use-default-roles \
--release-label emr-6.0.0 \
--ec2-attributes KeyName=spark \
--log-uri s3://p2-cs643-bucket/logs/  \
--ec2-attributes SubnetId=subnet-025c1de09afa7571b \
--instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge InstanceGroupType=CORE,InstanceCount=4,InstanceType=m5.xlarge \
--steps '[{"Args":["spark-submit","--deploy-mode","cluster","--class","com.njit.edu.cs643.v2.WinePredictionTraining","s3://p2-cs643-bucket/jars/wineprediction-1.0-7.jar","s3://p2-cs643-bucket/hubert/training.csv","s3://p2-cs643-bucket/hubert/output","5"],"Type":"CUSTOM_JAR","ActionOnFailure":"CANCEL_AND_WAIT","Jar":"command-runner.jar","Properties":"","Name":"Training"}, {"Args":["spark-submit","--deploy-mode","cluster","--class","com.njit.edu.cs643.v2.WinePredictionValidation","s3://p2-cs643-bucket/jars/wineprediction-1.0-7.jar","s3://p2-cs643-bucket/hubert/output/models","s3://p2-cs643-bucket/hubert/validation.csv","s3://p2-cs643-bucket/hubert/validation","5"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Prediction"}]' \
--auto-terminate


spark-submit --deploy-mode cluster --deploy-mode cluster --master yarn --class com.njit.edu.cs643.v2.WinePredictionTraining s3://p2-cs643-bucket/jars/wineprediction-1.0-2.jar s3://p2-cs643-bucket/hubert/training.csv s3://p2-cs643-bucket/hubert/output

spark-submit --deploy-mode cluster --deploy-mode cluster --master yarn --class com.njit.edu.cs643.v2.WinePredictionTraining s3://p2-cs643-bucket/jars/wineprediction-1.0-2.jar s3://p2-cs643-bucket/hubert/training.csv s3://p2-cs643-bucket/hubert/output


spark-submit --deploy-mode cluster --deploy-mode cluster --master yarn --class com.njit.edu.cs643.v2.WinePredictionValidation s3://p2-cs643-bucket/jars/wineprediction-1.0-3.jar s3://p2-cs643-bucket/hubert/output/models s3://p2-cs643-bucket/hubert/validation.csv s3://p2-cs643-bucket/hubert/validation

spark-submit --deploy-mode cluster --deploy-mode cluster --master yarn --class com.njit.edu.cs643.v2.WinePredictionValidation s3://p2-cs643-bucket/jars/wineprediction-1.0-3.jar s3://p2-cs643-bucket/hubert/output/models s3://p2-cs643-bucket/hubert/validation.csv s3://p2-cs643-bucket/hubert/validation




spark-submit --deploy-mode local  --class com.njit.edu.cs643.v2.WinePredictionTraining /opt/wineprediction-1.0-3.jar /opt/training.csv /opt/output



--deploy-mode client  --class com.njit.edu.cs643.v2.WinePredictionValidation /opt/spark/work-dir/wineprediction-1.0-4.jar /opt/spark/work-dir/models /opt/spark/work-dir/data/training.csv /opt/spark/work-dir/data/output

```bash
./spark-submit --deploy-mode client  --class com.njit.edu.cs643.v2.WinePredictionValidation <path to the jar>/wineprediction-1.0-5.jar <path to model>/models <path to data to evaluate>/data/training.csv <output folder>/output
```


```bash
docker run -it -v $(pwd)/data:/opt/spark/work-dir/data ml:test
```

docker run -it -v $(pwd)/data:/opt/spark/work-dir/data hubertnoyessie/cs643:1.4