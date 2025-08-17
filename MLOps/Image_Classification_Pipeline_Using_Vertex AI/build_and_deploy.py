
###Code to Build and Deploy the full pipeline
###This will be used in Cloud Builder

#Initialize pipeline object
from pipelines.train_pipeline import pipeline_controller
import time
import os

REGION="us-central1"
PROJECT_ID="qwiklabs-gcp-00-38ed10bd49be" 
BUCKET=f"{PROJECT_ID}-bucket"

PIPELINE_ROOT = f"gs://{BUCKET}/pipeline_root/"
DISPLAY_NAME = 'vertex-customml-pipeline{}'.format(str(int(time.time())))

print("Building pipeline {}".format(DISPLAY_NAME))

pipe = pipeline_controller(template_path="pipeline.json",
                           display_name="vertex-automlimage-classif", 
                           pipeline_root=PIPELINE_ROOT,
                           project_id=PROJECT_ID,
                           region=REGION)

#Build and Compile pipeline
pipe._build_compile_pipeline()

##Submit Job
pipe._submit_job()
