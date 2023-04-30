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
--steps Type=Spark,Name=”CS643 Spark cluster”,ActionOnFailure=CONTINUE,Args=[ — class,com.jeanr84.sparkjob.SparkJob,s3://my-second-emr-bucket/tutorialEMR/spark-job-0.0.1-SNAPSHOT.jar,s3://my-second-emr-bucket/tutorialEMR/input.txt,s3://my-second-emr-bucket/tutorialEMR/output] \
--use-default-roles \
--ec2-attributes SubnetId=subnet-e7ba9d9c --instance-type m4.large --instance-count 3 \
--auto-terminate


aws emr create-cluster --name "CS643 Spark cluster" --release-label emr-6.0.0 --applications Name=Spark \
--ec2-attributes KeyName=spark --instance-type m4.large --instance-count 3 --use-default-roles