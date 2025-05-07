import os
import argparse
from nipype.interfaces.fsl.maths import MathsCommand, MultiImageMaths
from nipype import Node, Workflow
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import Merge
import nipype.interfaces.utility as util  # utility


def run():
    
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--study_path', 
                        type=str, 
                        help='Path to the study data',
                        default='/media/robbis/DATA/fmri/optical_nerve/dev/data/orig/'
                        )
    parser.add_argument('--subject_list', 
                        type=str, 
                        help='List of subjects to process',
                        default=['001', '002']
                        )
    
    args = parser.parse_args()
    
    study_path = args.study_path
    subject_list = args.subject_list
    
    # Create a DataGrabber node
    dg = Node(DataGrabber(
                infields=['subject_id'], 
                outfields=['otr', 'otl', 'onc', 'onr', 'onl']),
                name='datagrabber')
    dg.inputs.base_directory = study_path
    dg.inputs.template = '%d/*'
    dg.inputs.field_template = dict(otr='%s/otr.nii.gz',
                                    otl='%s/otl.nii.gz', 
                                    onc='%s/onc.nii.gz', 
                                    onr='%s/onr.nii.gz', 
                                    onl='%s/onl.nii.gz')
    dg.inputs.template_args = dict(otr=[['subject_id']], 
                                otl=[['subject_id']],
                                onc=[['subject_id']],
                                onr=[['subject_id']], 
                                onl=[['subject_id']])
    dg.inputs.subject_id = subject_list
    dg.inputs.sort_filelist = True

    infosource = Node(
        interface=util.IdentityInterface(fields=['subject_id']), 
        name="infosource")
    infosource.iterables = ('subject_id', subject_list)

    # Create a DataSink node
    ds = Node(DataSink(base_directory=os.path.join('/media/robbis/DATA/fmri/optical_nerve/dev/data/', 'proc')),
                    name='datasink')

    ds.inputs.substitutions = [('container', ''),
                               ('segment', ''),
                               ('_subject_id_', '')]

    workflow = Workflow(name='aVP_prep_workflow', base_dir=study_path)  # Update base_dir accordingly

    # Define the processing nodes
    ot_math_r =     Node(MathsCommand(args='-thr 10 -uthr 10 -bin -mul 16', out_file='ot_r.nii.gz'), 
                        name='ot_math_r')
    ot_math_l =     Node(MathsCommand(args='-thr 10 -uthr 10 -bin -mul 16', out_file='ot_l.nii.gz'), 
                        name='ot_math_l')
    oc_r_math =     Node(MathsCommand(args='-thr 8 -uthr 8 -bin -mul 8', out_file='oc_r.nii.gz'), 
                        name='oc_r_math')
    oc_l_math =     Node(MathsCommand(args='-thr 9 -uthr 9 -bin -mul 8', out_file='oc_l.nii.gz'), 
                        name='oc_l_math')
    oninor_math_r = Node(MathsCommand(args='-thr 2 -uthr 2 -bin', out_file='oninor_r.nii.gz'), 
                        name='oninor_math_r')
    oninor_math_l = Node(MathsCommand(args='-thr 2 -uthr 2 -bin', out_file='oninor_l.nii.gz'), 
                        name='oninor_math_l')
    oninca_math_r = Node(MathsCommand(args='-thr 4 -uthr 4 -bin -mul 2', out_file='oninca_r.nii.gz'), 
                        name='oninca_math_r')
    oninca_math_l = Node(MathsCommand(args='-thr 4 -uthr 4 -bin -mul 2', out_file='oninca_l.nii.gz'), 
                        name='oninca_math_l')
    onincr_math_r = Node(MathsCommand(args='-thr 6 -uthr 6 -bin -mul 4', out_file='onincr_r.nii.gz'), 
                        name='onincr_math_r')
    onincr_math_l = Node(MathsCommand(args='-thr 6 -uthr 6 -bin -mul 4', out_file='onincr_l.nii.gz'), 
                        name='onincr_math_l')


    # Connect the DataGrabber outputs to the processing nodes
    workflow.connect([
        (infosource, dg, [('subject_id', 'subject_id')]),
        (infosource, ds, [('subject_id', 'container')]),
        (dg, ot_math_r, [('otr', 'in_file')]),
        (dg, ot_math_l, [('otl', 'in_file')]),
        (dg, oc_r_math, [('onc', 'in_file')]),
        (dg, oc_l_math, [('onc', 'in_file')]),
        (dg, oninor_math_r, [('onr', 'in_file')]),
        (dg, oninor_math_l, [('onl', 'in_file')]),
        (dg, oninca_math_r, [('onr', 'in_file')]),
        (dg, oninca_math_l, [('onl', 'in_file')]),
        (dg, onincr_math_r, [('onr', 'in_file')]),
        (dg, onincr_math_l, [('onl', 'in_file')])
    ])

    # Connect the processing nodes to the DataSink
    workflow.connect([
        (ot_math_r, ds, [('out_file',       'segment.@ot_r')]),
        (ot_math_l, ds, [('out_file',       'segment.@ot_l')]),
        (oc_r_math, ds, [('out_file',       'segment.@oc_r')]),
        (oc_l_math, ds, [('out_file',       'segment.@oc_l')]),
        (oninor_math_r, ds, [('out_file',   'segment.@oninor_r')]),
        (oninor_math_l, ds, [('out_file',   'segment.@oninor_l')]),
        (oninca_math_r, ds, [('out_file',   'segment.@oninca_r')]),
        (oninca_math_l, ds, [('out_file',   'segment.@oninca_l')]),
        (onincr_math_r, ds, [('out_file',   'segment.@onincr_r')]),
        (onincr_math_l, ds, [('out_file',   'segment.@onincr_l')])
    ])

    # Define the final addition operations

    operand_files_r = Node(Merge(5), name='operand_files_r')
    operand_files_l = Node(Merge(5), name='operand_files_l')

    workflow.connect([
        (oc_r_math, operand_files_r, [('out_file', 'in1')]),
        (oninor_math_r, operand_files_r, [('out_file', 'in2')]),
        (oninca_math_r, operand_files_r, [('out_file', 'in3')]),
        (onincr_math_r, operand_files_r, [('out_file', 'in4')]),
        (oc_l_math, operand_files_l, [('out_file', 'in1')]),
        (oninor_math_l, operand_files_l, [('out_file', 'in2')]),
        (oninca_math_l, operand_files_l, [('out_file', 'in3')]),
        (onincr_math_l, operand_files_l, [('out_file', 'in4')])
    ])


    on_math_r = Node(MultiImageMaths(out_file='on_r.nii.gz'), name='on_math_r')
    on_math_r.inputs.op_string = '-add %s -add %s -add %s -add %s'

    on_math_l = Node(MultiImageMaths(out_file='on_l.nii.gz'), name='on_math_l')
    on_math_l.inputs.op_string = '-add %s -add %s -add %s -add %s'

    # Connect the final addition operations to the MultiImageMaths nodes
    workflow.connect([
        (ot_math_l, on_math_l, [('out_file', 'in_file')]),
        (ot_math_r, on_math_r, [('out_file', 'in_file')]),
        (operand_files_r, on_math_r, [('out', 'operand_files')]),
        (operand_files_l, on_math_l, [('out', 'operand_files')])
    ])


    workflow.connect([
        (on_math_r, ds, [('out_file', 'segment.@on_r')]),
        (on_math_l, ds, [('out_file', 'segment.@on_l')]),
    ])

    workflow.run()

if __name__ == '__main__':
    run()