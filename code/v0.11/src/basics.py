import os
from nipype.interfaces.fsl import ImageStats
from nipype import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink
import pandas as pd

StudyPath = '/path/to/StudyPath'  # Update this path accordingly

# Create a DataGrabber node
dg = Node(DataGrabber(infields=['subject_id'], outfields=['files']),
          name='datagrabber')
dg.inputs.base_directory = os.path.join(StudyPath, 'data', 'proc')
dg.inputs.template = '*'
dg.inputs.field_template = dict(files='%s/*')
dg.inputs.template_args = dict(files=[['subject_id']])
dg.inputs.sort_filelist = True

# Create a DataSink node
ds = Node(DataSink(base_directory=os.path.join(StudyPath, 'results')),
          name='datasink')

# Initialize the output CSV file
outFile = os.path.join(StudyPath, 'results', 'volume_orig_20230708.csv')
if not os.path.exists(os.path.dirname(outFile)):
    os.makedirs(os.path.dirname(outFile))
with open(outFile, 'w') as f:
    f.write("Subject;NerveSegment;Side;NumberVoxels;Volume\n")

# Define the processing nodes
def compute_volume(file_path):
    stats = ImageStats(in_file=file_path, op_string='-V')
    result = stats.run()
    num_voxels, volume = result.outputs.out_stat
    return num_voxels, volume

def process_subject(subject_id, files):
    results = []
    for file_path in files:
        if any(seg in file_path for seg in ['ot', 'oc', 'onincr', 'oninca', 'oninor']):
            num_voxels, volume = compute_volume(file_path)
            nerve_segment = os.path.basename(file_path).split('_')[0]
            side = os.path.basename(file_path).split('_')[1][0]
            results.append([subject_id, nerve_segment, side, num_voxels, volume])
    return results

# Create the workflow
workflow = Workflow(name='aVP_basics_workflow', base_dir='/path/to/workflow_dir')  # Update base_dir accordingly

# Define the processing function
def process_and_save(subject_id, files):
    results = process_subject(subject_id, files)
    df = pd.DataFrame(results, columns=["Subject", "NerveSegment", "Side", "NumberVoxels", "Volume"])
    df.to_csv(outFile, mode='a', header=False, index=False)

# Create a Node for the processing function
process_node = Node(name='process_node', interface=Function(input_names=['subject_id', 'files'], output_names=[], function=process_and_save))

# Connect the DataGrabber outputs to the processing node
workflow.connect([
    (dg, process_node, [('files', 'files')]),
    (dg, process_node, [('subject_id', 'subject_id')])
])

workflow.run()