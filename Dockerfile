FROM apache/spark

COPY target/wineprediction-*.jar /opt/spark/work-dir/wineprediction.jar
COPY models /opt/spark/work-dir/models

ENTRYPOINT ["/bin/bash", "-c" , "/opt/spark/bin/spark-submit  --deploy-mode client --class com.njit.edu.cs643.v2.WinePredictionValidation /opt/spark/work-dir/wineprediction.jar  /opt/spark/work-dir/models /opt/spark/work-dir/data/data.csv /opt/spark/work-dir/data/output"]
