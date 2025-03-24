import os
from nipype.interfaces.fsl import MathsCommand, FLIRT, ImageMaths, ImageStats, ImageResample
from nipype import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink

StudyPath = '/path/to/StudyPath'  # Update this path accordingly
imPath = '/path/to/imPath'  # Update this path accordingly
anat = 'anat'  # Update this variable accordingly
baseImage = 'baseImage'  # Update this variable accordingly

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

# Create the workflow
workflow = Workflow(name='aVP_resample_workflow', base_dir='/path/to/workflow_dir')  # Update base_dir accordingly

# Define the processing nodes
def process_image(file_path):
    nn = os.path.basename(file_path).replace('.nii.gz', '')
    mm = os.path.dirname(file_path)

    # fslmaths ${ii} -mul 1 ${mm}/${nn}2.nii.gz
    mul_node = Node(MathsCommand(in_file=file_path, op_string='-mul 1', out_file=os.path.join(mm, f'{nn}2.nii.gz')), name='mul_node')
    
    # fslorient -setqformcode 1 ${mm}/${nn}2.nii.gz
    set_qform_node = Node(ImageMaths(in_file=os.path.join(mm, f'{nn}2.nii.gz'), op_string='-setqformcode 1'), name='set_qform_node')
    
    # fslorient -setsformcode 1 ${mm}/${nn}2.nii.gz
    set_sform_node = Node(ImageMaths(in_file=os.path.join(mm, f'{nn}2.nii.gz'), op_string='-setsformcode 1'), name='set_sform_node')
    
    # fslhd -x ${mm}/${nn}2.nii.gz | sed "s/dy = '[^\']*'/dy = '0.0245'/g" > ${mm}/hd.xml
    hd_node = Node(ImageStats(in_file=os.path.join(mm, f'{nn}2.nii.gz'), op_string='-x'), name='hd_node')
    
    # fslcreatehd ${mm}/hd.xml ${mm}/${nn}2.nii.gz
    create_hd_node = Node(ImageResample(in_file=os.path.join(mm, f'{nn}2.nii.gz'), out_file=os.path.join(mm, f'{nn}2.nii.gz')), name='create_hd_node')
    
    # fslorient -copyqform2sform ${mm}/${nn}2.nii.gz
    copy_qform_node = Node(ImageMaths(in_file=os.path.join(mm, f'{nn}2.nii.gz'), op_string='-copyqform2sform'), name='copy_qform_node')
    
    # flirt -in ${mm}/${nn}2.nii.gz -ref ${mm}/${nn}2.nii.gz -applyisoxfm 0.6 -datatype int -interp nearestneighbour -nosearch -out ${mm}/${nn}_iso06pre
    flirt_node = Node(FLIRT(in_file=os.path.join(mm, f'{nn}2.nii.gz'), reference=os.path.join(mm, f'{nn}2.nii.gz'), apply_isoxfm=0.6, datatype='int', interp='nearestneighbour', out_file=os.path.join(mm, f'{nn}_iso06pre.nii.gz')), name='flirt_node')
    
    # imcp ${mm}/${nn}_iso06pre ${mm}/${nn}_iso06
    imcp_node = Node(ImageMaths(in_file=os.path.join(mm, f'{nn}_iso06pre.nii.gz'), op_string='-copy', out_file=os.path.join(mm, f'{nn}_iso06.nii.gz')), name='imcp_node')
    
    # fslorient -setsform -0.6 0 0 74.4 0 0.6 0 -60.6  0 0 0.6 -21.0 0 0 0 1 ${mm}/${nn}_iso06
    set_sform_node2 = Node(ImageMaths(in_file=os.path.join(mm, f'{nn}_iso06.nii.gz'), op_string='-setsform -0.6 0 0 74.4 0 0.6 0 -60.6  0 0 0.6 -21.0 0 0 0 1'), name='set_sform_node2')
    
    # fslorient -setqform -0.6 0 0 74.4 0 0.6 0 -60.6  0 0 0.6 -21.0 0 0 0 1 ${mm}/${nn}_iso06
    set_qform_node2 = Node(ImageMaths(in_file=os.path.join(mm, f'{nn}_iso06.nii.gz'), op_string='-setqform -0.6 0 0 74.4 0 0.6 0 -60.6  0 0 0.6 -21.0 0 0 0 1'), name='set_qform_node2')
    
    # Remove intermediate files
    os.remove(os.path.join(mm, f'{nn}_iso06pre.nii.gz'))
    os.remove(os.path.join(mm, f'{nn}2.nii.gz'))

    return os.path.join(mm, f'{nn}_iso06.nii.gz')

# Create a Node for the processing function
process_node = Node(name='process_node', interface=Function(input_names=['file_path'], output_names=['out_file'], function=process_image))

# Connect the DataGrabber outputs to the processing node
workflow.connect([
    (dg, process_node, [('files', 'file_path')])
])

# Connect the processing node to the DataSink
workflow.connect([
    (process_node, ds, [('out_file', '@out_file')])
])

workflow.run()