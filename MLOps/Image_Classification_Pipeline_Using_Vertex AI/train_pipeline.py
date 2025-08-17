
import os

import kfp
import time
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import NamedTuple, Dict

from google.cloud import aiplatform

#Main pipeline class
class pipeline_controller():
    def __init__(self, template_path, display_name, pipeline_root, project_id, region):
        self.template_path = template_path
        self.display_name = display_name
        self.pipeline_root = pipeline_root
        self.project_id = project_id
        self.region = region
    
    def _build_compile_pipeline(self):
        """Method to build and compile pipeline"""
        self.pipeline = self._get_pipeline(self.project_id, self.region)
        compiler.Compiler().compile(
            pipeline_func=self.pipeline, package_path=self.template_path
        )
        
    def _submit_job(self):
        """Method to Submit ML Pipeline job"""
        #Next, define the job:
        ml_pipeline_job = aiplatform.PipelineJob(
            display_name=self.display_name,
            template_path=self.template_path,
            pipeline_root=self.pipeline_root,
            project=self.project_id,
            location=self.region,
            # parameter_values={"project": self.project_id, "display_name": self.display_name},
            enable_caching=False
        )

        #And finally, run the job:
        ml_pipeline_job.submit()
    
    def _get_pipeline(self, PROJECT_ID, REGION):
        ## Light weight component to create an Image DS
        @component(
            base_image="python:3.9-slim",
            packages_to_install=["google-api-core==2.10.2", "google-cloud", "google-cloud-aiplatform", "typing", "kfp"],
        )
        def create_ds(project: str, 
                      display_name: str, 
                      gcs_source: str, 
                      import_schema_uri: str, 
                      timeout: int, 
                      dataset: Output[Dataset]):

            from google.cloud import aiplatform
            from google.cloud.aiplatform import datasets
            from kfp.v2.dsl import Dataset

            aiplatform.init(project=project)

            obj_dataset = datasets.ImageDataset.create(
                display_name=display_name,
                gcs_source=gcs_source,
                import_schema_uri=import_schema_uri,
                create_request_timeout=timeout,
            )

            obj_dataset.wait()

            dataset.uri = obj_dataset.gca_resource.name
            dataset.metadata = {
                'resourceName': obj_dataset.gca_resource.name
            }
        
        """Main method to Create pipeline"""
        @pipeline(name=self.display_name,
                    pipeline_root=self.pipeline_root)
        def pipeline_fn(
            project: str = PROJECT_ID, 
            region: str = REGION
        ):
            
            from google_cloud_pipeline_components import aiplatform as gcc_aip
            from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp, ModelDeployOp)
            import google.cloud.aiplatform as aip

            ds_op = create_ds(
                project=project,
                display_name="flowers",
                gcs_source="gs://cloud-samples-data/vision/automl_classification/flowers/all_data_v2.csv",
                import_schema_uri=aip.schema.dataset.ioformat.image.single_label_classification,
                timeout=3600
            )

            training_job_run_op = gcc_aip.AutoMLImageTrainingJobRunOp(
                project=project,
                location=region,
                display_name="train-automl-flowers",
                prediction_type="classification",
                model_type="CLOUD",
                dataset=ds_op.outputs["dataset"].ignore_type(),
                model_display_name="train-automl-flowers",
                training_fraction_split=0.6,
                validation_fraction_split=0.2,
                test_fraction_split=0.2,
                budget_milli_node_hours=8000,
            )

            endpoint_op = EndpointCreateOp(
                project=project,
                location=region,
                display_name="train-automl-flowers",
            )

            ModelDeployOp(
                model=training_job_run_op.outputs["model"],
                endpoint=endpoint_op.outputs["endpoint"],
                automatic_resources_min_replica_count=1,
                automatic_resources_max_replica_count=1,
            )

        #Returns as the output of _get_pipeline()
        return pipeline_fn
