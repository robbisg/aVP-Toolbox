
import avpy.main as avpy_main
import sys
from joblib import Parallel, delayed
from nilearn import plotting

data_paths = [
    "/media/robbis/DATA/fmri/optical_nerve/data/CISS_Manual_Segmentations_HC+PTS/HC/", 
    "/media/robbis/DATA/fmri/optical_nerve/data/CISS_Manual_Segmentations_HC+PTS/PTS/",
    #'/media/robbis/DATA/fmri/optical_nerve/data/ERR/',
    #"/media/robbis/DATA/fmri/optical_nerve/dati+risultati/HC/",
    #"/media/robbis/DATA/fmri/optical_nerve/dati+risultati/MS/",
    #"/home/robbis/git/aVP-Toolbox/data/error/",
    #"/media/robbis/DATA/fmri/optical_nerve/data/AI/seg_ciss_0.6/test/",
    #"/media/robbis/DATA/fmri/optical_nerve/data/AI/seg_ciss_0.6/StudyFolder/",
    #"/media/robbis/DATA/fmri/optical_nerve/data/AI/seg_ciss_0.6/StudyFolderGT/",
    #"/media/robbis/DATA/fmri/optical_nerve/data/AI/seg_ciss_0.6/StudyFolderR1/",
    #"/media/robbis/DATA/fmri/optical_nerve/data/AI/seg_ciss_0.6/StudyFolderR2/",
]

for data_path in data_paths:
    
    sys.argv = ["avpy", "--root-dir", data_path]
    avpy_main.main()
    


# Create a report using mne Report to look at template images
from mne import Report
import os
from nilearn import plotting

report = Report(title="aVP Toolbox Report", verbose=True)

for data_path in data_paths:
    
    # Add images from the templates directory
    template_file = os.path.join(data_path, "templates", "aVP_prob.nii.gz")
    fig = plotting.plot_img(template_file)
    
    template_fig = os.path.join(data_path, "templates", "aVP_prob.png")
    fig.savefig(template_fig)
    
    report.add_image(
        image=template_fig,
        section=f"Template Image for {data_path}",
        title="Template Image",
        caption="Template atlas",
    )
report.save("avpy_report.html", open_browser=True, overwrite=True)