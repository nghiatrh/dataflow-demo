import apache_beam as beam
import logging
from joblib import load
import numpy as np
import pandas as pd

from google.cloud import storage
from apache_beam.options.pipeline_options import StandardOptions, GoogleCloudOptions, SetupOptions, PipelineOptions

from sklearn.ensemble import RandomForestClassifier

#setup global
dummy_dict = {'Yes': 1, 'No': 0}
internet_dict = {'No': 0, 'No internet service': 1, 'Yes': 2}
yesno_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
# =============================================================================
# Build and run the pipeline
# =============================================================================
def run(argv=None):
    pipeline_options = PipelineOptions(flags=argv)

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = 'your-project' #change this
    google_cloud_options.job_name = 'telco-churn-prediction'
    google_cloud_options.staging_location = 'gs://path/to/your/staging' #change this
    google_cloud_options.temp_location = 'gs://path/to/your/temp' #change this
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(SetupOptions).setup_file = "./setup.py"
    logging.info("Pipeline arguments: {}".format(pipeline_options))

    # table_schema = 'customerID: STRING, prediction: FLOAT'
    query = ('select * from `your-project.Telco_Churn.input`')
    bq_source = beam.io.BigQuerySource(query=query, use_standard_sql=True)
    p = beam.Pipeline(options=pipeline_options)
    (p
     | "Read data from BQ" >> beam.io.Read(bq_source)
     | "Preprocess data" >> beam.ParDo(FormatInput())
     | "predicting" >> beam.ParDo(
                PredictSklearn(project='your-project', bucket_name='your-bucket-name', model_path='/path/to/model_rf.joblib',
                               destination_name='model_rf.joblib'))
     | "Write data to BQ" >> beam.io.WriteToBigQuery(table='prediction', dataset='Telco_Churn', project='your-project',
                                                     # schema=table_schema,
                                                     # create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                                                     write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
     )

    result = p.run()
    result.wait_until_finish()


# =============================================================================
# Function to download model from bucket
# =============================================================================
def download_blob(bucket_name=None, source_blob_name=None, project=None, destination_file_name=None):
    storage_client = storage.Client(project)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


class FormatInput(beam.DoFn):
    def process(self, element):
        """ Format the input to the desired shape"""
        df = pd.DataFrame([element], columns=element.keys())
        df[yesno_cols] = df[yesno_cols].apply(lambda x: x.map(dummy_dict))
        df[internet_cols] = df[internet_cols].apply(lambda x: x.map(internet_dict))
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
        df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'No phone service': 1, 'Yes': 2})
        df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
        df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
        df['PaymentMethod'] = df['PaymentMethod'].map(
            {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3})
        output = df.to_dict('records')
        return output #return a dict for easier comprehension

class PredictSklearn(beam.DoFn):

    def __init__(self, project=None, bucket_name=None, model_path=None, destination_name=None):
        self._model = None
        self._project = project
        self._bucket_name = bucket_name
        self._model_path = model_path
        self._destination_name = destination_name

    def setup(self):
        """Download sklearn model from GCS"""
        logging.info(
            "Sklearn model initialisation {}".format(self._model_path))
        download_blob(bucket_name=self._bucket_name, source_blob_name=self._model_path,
                      project=self._project, destination_file_name=self._destination_name)
        # unpickle sklearn model
        self._model = load(self._destination_name)

    def process(self, element):
        """Predicting using developed model"""
        input_dat = {k: element[k] for k in element.keys() if k not in ['customerID']}
        tmp = np.array(list(i for i in input_dat.values()))
        tmp = tmp.reshape(1, -1)
        element["prediction"] = self._model.predict_proba(tmp)[:,1].item()
        output = {k: element[k] for k in element.keys() if k in ['customerID', 'prediction']}
        output['customerID'] = str(output['customerID'])
        return [output]


# log the output
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()