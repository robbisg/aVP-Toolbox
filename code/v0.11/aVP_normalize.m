%  
% This script is part of the aVP-Toolbox v0.11 - 2023 software. 
%
% aVP-Toolbox ("The software") is licensed under the Creative Commons Attribution 4.0 International License, 
% permitting use, sharing, adaptation, distribution and reproduction in any medium or format, 
% as long as you give appropriate credit to the original author(s) and the source, provide 
% a link to the Creative Commons licence, and indicate if changes were made. 
% The licensor offers the Licensed Material as-is and as-available, and makes no 
% representations or warranties of any kind concerning the Licensed Material, 
% whether express, implied, statutory, or other. This includes, without limitation, 
% warranties of title, merchantability, fitness for a particular purpose, non-infringement, 
% absence of latent or other defects, accuracy, or the presence or absence of errors, 
% whether or not known or discoverable. Where disclaimers of warranties are not allowed 
% in full or in part, this disclaimer may not apply to You. 
% Please go to http://creativecommons.org/licenses/by/4.0/ to view a complete copy of this licence.
%
% reads in:
% - nifti files of
%    axial segmentation masks:   $inPATH/SubjectID/on_SIDE_nii.gz
%
% calls 'aVP_resample.sh'
%
% produces: 
% - nifti files 10-fold expanded in AP (y) direction 
%    straightened (centered) aVP segments:   _linearize_4.nii.gz
%    normalized (interpolated for length conservation) aVP segments: _normalized_4.nii.gz
% - matlab mat file of the normalized aVP segments: _normalized.mat
% - text file listings of
%    data values in each slice
%    holes 
%    range
%    length, CSA in each aVP anatomical section
% to be run from code folder of StudyFolder

curwd=pwd();
fileID = fopen(strcat(curwd,'/ONcontrol.txt'),'r');
StudyPath=fscanf(fileID,'%s\n');
fclose(fileID);

inPath=strcat(StudyPath,'/data/proc');
outImPath=strcat(StudyPath,'/data/proc');
outResPath=strcat(StudyPath,'/results');

addpath(genpath(strcat(curwd,'/localtoolboxes/20130227_xlwrite')));
addpath(genpath(strcat(curwd,'/localtoolboxes/nifti_tools-master')));

javaaddpath(strcat(curwd,'/localtoolboxes/20130227_xlwrite/poi_library/poi-3.8-20120326.jar'));
javaaddpath(strcat(curwd,'/localtoolboxes/20130227_xlwrite/poi_library/poi-ooxml-3.8-20120326.jar'));
javaaddpath(strcat(curwd,'/localtoolboxes/20130227_xlwrite/poi_library/poi-ooxml-schemas-3.8-20120326.jar'));
javaaddpath(strcat(curwd,'/localtoolboxes/20130227_xlwrite/poi_library/xmlbeans-2.3.0.jar'));
javaaddpath(strcat(curwd,'/localtoolboxes/20130227_xlwrite/poi_library/dom4j-1.6.1.jar'));
javaaddpath(strcat(curwd,'/localtoolboxes/20130227_xlwrite/poi_library/stax-api-1.0.1.jar'));

% name some output files and their header lines

DataFile=strcat(outResPath,'/aVP_slice_data.xlsx');                     
HoleFile=strcat(outResPath,'/log_check_hole.xlsx');
RangeFile=strcat(outResPath,'/log_check_range.xlsx');
LenStretchFile=strcat(outResPath,'/aVP_section_CSA_length.xlsx');

tablabels=[ {'curr_sli_yz'}, ...  % 1
            {'orig_sli_yz'}, ...  % 2
            {'point_y'}, ...      % 3
            {'point_z'}, ...      % 4
            {'circshift_y'}, ...  % 5
            {'circshift_z'}, ...  % 6
            {'dist'}, ...         % 7
            {'int_dist_x10'}, ... % 8
            {'len_on'}, ...       % 9
            {'tot_len'}, ...      % 10
            {'mMax'}, ...         % 11
            {'save_len'}, ...     % 12
            {'CSArea'}, ...       % 13
            {'Eccent'}, ...       % 14
            {'MajAxis'}, ...      % 15
            {'MinAxis'}, ...      % 16
            {'AvgCSA'} ];         % 17




stretxt=[ {'Subject'} {'ONsection'} {'side'} {'TotLength'} {'OT_length'} {'OC_length'} {'iCran_length'} {'iCan_length'} {'iOrb_length'} {'OT_CSA'} {'OC_CSA'} {'iCran_CSA'} {'iCan_CSA'} {'iOrb_CSA'} {'SegmCode 1'} {'SegmCode 2'} {'SegmCode 3'} {'SegmCode 4'} {'SegmCode 5'}];   

rangetxt=[ {'Subject'} {'ONsection'} {'side'} {'Slice...'} ];
xlRange='A1';
xlwrite(RangeFile,rangetxt,'Sheet 1', xlRange);

holetxt=[ {'Subject'} {'ONsection'} {'side'} {'Slice...'} ];
xlRange='A1';
xlwrite(HoleFile,holetxt,'Sheet 1', xlRange);

% name the output images 
Lin4image='_linearize_4bc.nii.gz';
Lin4mat='_linearize_4bc.mat';

Norm4image='_normalized_4bc.nii.gz';
Norm4mat='_normalized_4bc.mat';

fullNorm4image='_full_normalized_4bc.nii.gz';

sheet=1;
sides={'l','r' };
isbj=0;
tablelength=[];

maxNslices=2500;   
ismask=0; % 0: use the combination of aVP segments. 1: use the individual aVP segments (not tested)

fileID = fopen(strcat(StudyPath,'/data/sbj.list'));
subject_list = textscan(fileID,'%s');
fclose(fileID);

resolution_increase = 10;

for sbj = subject_list{1}'

    isbj=isbj+1;

    for side=1:2 ;     

        clear cc_value
        fname=strcat(inPath,'/',sbj{1},'/on_',sides{side});

        disp(strcat('INFO: Analyzing ', sbj{1},' - on_',sides{side}));

        if not(exist(strcat(fname,'.nii.gz')))
            disp(strcat('could not find: ',fname,'.nii.gz'))
            return;
        end
        
        % take into account that the image considers the origin to be lower left, but matlab upper left. 
        aswap = load_nifti(fname);
        aa = aswap;
        aa.img = flip(aswap.img,1);
        oo = aa;
        
        % will deal with coronal cuts through the data, so x,z (d1, d2) is the plane, and y is the slice (d3) ...
        x_dim=size(aa.img,1);
        z_dim=size(aa.img,3);
        y_dim=size(aa.img,2);
        
        x_resolution=aa.hdr.dime.pixdim(2);
        z_resolution=aa.hdr.dime.pixdim(4);
        y_resolution=aa.hdr.dime.pixdim(3); 
        
        % calcualte the center of the slice in order to be able to shift the centroid of the ROI there. 
        image_center=[round(z_dim/2) round(x_dim/2)];
        active_slice_counter=0;
        segment_type = 0;
        
        table=[];
        
        current_max_value=0;
        
        for y=1:y_dim % advance along y 
            selected_y_slice = squeeze(aa.img(:,y,:)); % take ith xz "slice", eliminating the other ys 
            
            max_voxel_value = max(max(selected_y_slice));
            
            if max_voxel_value > 0 % if max in slice not 0, there is someting in the "slice" it is active
                
                % use flag as counter of active slices 
                % use flagC to record changes in ON segment
                if active_slice_counter == 0, 
                    segment_type = 1; 
                    current_max_value = max_voxel_value;
                else
                    if not( current_max_value == max_voxel_value ) ,
                        segment_type = segment_type + 1 ;
                    end
                end

                active_slice_counter = active_slice_counter + 1;
                
                cc_value{active_slice_counter}.current_slice_yz=active_slice_counter;
                cc_value{active_slice_counter}.original_slice_yz=y;       
                cc_value{active_slice_counter}.flagC = segment_type;
                cc_value{active_slice_counter}.mm = max_voxel_value ;

                % Deal with centering the mask in a new version of the image           
                
                binarized_slice = selected_y_slice > 0; % binarize
                st = regionprops( binarized_slice, 'centroid' ); % find the centroid
                area0 = regionprops( binarized_slice, 'area' );
                temp_centroids = cat(1, st.Centroid);
                orig_centroids = temp_centroids(1,:); 
                
                slice_properties = regionprops(binarized_slice,'centroid','Area','MajorAxisLength','MinorAxisLength','Eccentricity');

                cc_value{active_slice_counter}.majaxis = slice_properties(1).MajorAxisLength*x_resolution ;
                cc_value{active_slice_counter}.minaxis = slice_properties(1).MinorAxisLength*z_resolution ;
                cc_value{active_slice_counter}.area = slice_properties(1).Area*x_resolution*z_resolution ;
                cc_value{active_slice_counter}.eccent = slice_properties(1).Eccentricity ;
                    
                % shift the centroid of the region to the center of the image
                
                x_center_shift = image_center(2) - round(orig_centroids(2));
                z_center_shift = image_center(1) - round(orig_centroids(1));

                cc = circshift(selected_y_slice, [ x_center_shift z_center_shift ]);
                cc_value{active_slice_counter}.circshift = [ x_center_shift z_center_shift ];
              
                % saves the mask (imask=1) or the segments
                if ismask == 0
                    oo.img(:,y,:) = cc;
                else
                    oo.img(:,y,:) = sign(cc);
                end
                
                % the above eliminates the contribution to length associated with obliqueness of the 
                % optic nerve. 
                % To account for this, interpolate 10-fold via replication, then add or
                % remove interpolated slices to attain the ON length seen in the
                % original

                length_on = 0;
                cc_value{active_slice_counter}.point = orig_centroids; %is it the same of temp_centroids?
                cc_value{active_slice_counter}.point_original = temp_centroids;

                cc_value{active_slice_counter}.distance = 0;
                cc_value{active_slice_counter}.length_on = length_on;

                cc_value{active_slice_counter}.slice = oo.img(:,y,:);

                % replicate the slice 10 times
                for kk=1:10, 
                    cc_value{active_slice_counter}.vol(1:x_dim,kk,1:z_dim) = reshape(oo.img(:,y,:),[x_dim 1 z_dim]) ; 
                end

                % create an empty slice and start values for lengths
                cc_value{active_slice_counter}.intra(1:x_dim,1,1:z_dim) = reshape(oo.img(:,y,:),[x_dim 1 z_dim]) * 0 ;                    
                cc_value{active_slice_counter}.length_on=cc_value{active_slice_counter}.length_on+y_resolution;
                cc_value{active_slice_counter}.total_length = cc_value{active_slice_counter}.length_on;
                cc_value{active_slice_counter}.save_length = 0;       
                cc_value{active_slice_counter}.int_distance_x10 = 0;
                cc_value{active_slice_counter}.avgCSA = 0;  

                if active_slice_counter > 1
                    % calculate the distance between original centroids
                    % between this and neigbouring slice

                    previous_slice = active_slice_counter - 1;

                    z_center_displacement = (cc_value{active_slice_counter}.point(:,1) - cc_value{previous_slice}.point(:,1));
                    x_center_displacement = (cc_value{active_slice_counter}.point(:,2) - cc_value{previous_slice}.point(:,2));


                    zz = z_resolution * z_center_displacement;
                    xx = x_resolution * x_center_displacement;
                    yy = y_resolution * 2;                             % why times 2 ?
                    cc_value{active_slice_counter}.distance = sqrt(xx*xx + yy*yy + zz*zz);

                    % multiply by 10 and round to integer
                    distance_gap = round((cc_value{active_slice_counter}.distance - 2*y_resolution) / y_resolution * 10);   % why times 2 ?
                    % if 0, centroids were already aligned, and no
                    % correction is needed,

                    %if negative something is strange - ignore it.
                    if distance_gap < 0
                        current_slice_print = y;
                        previous_slice_print = cc_value{active_slice_counter-1}.original_slice_yz;
                        disp(strcat('WARNING: Negative distance between slice', num2str(current_slice_print), num2str(previous_slice_print)));

                        distance_gap = 0;
                    end

                    % if positive make insertions.
                    if distance_gap > 0
                        pp = 0;
                        % for first half of interpolates use prior slice 
                        % for the second half, use the current slice
                        for kk=1:distance_gap
                            if kk < fix(distance_gap / 2),
                                    cc_value{active_slice_counter}.intra(1:end,kk,1:end) = cc_value{active_slice_counter-1}.vol(1:end,1,1:end);
                            else
                                    cc_value{active_slice_counter}.intra(1:end,kk,1:end) = cc_value{active_slice_counter}.vol(1:end,1,1:end); 
                            end
                        end
                    end

                    % update length of ON
                    cc_value{active_slice_counter}.length_on = cc_value{active_slice_counter}.length_on + distance_gap/10 * y_resolution; 

                    % and total length.
                    cc_value{active_slice_counter}.total_length = cc_value{active_slice_counter-1}.total_length + cc_value{active_slice_counter}.length_on ;
                    cc_value{active_slice_counter}.int_distance_x10 = distance_gap;

                    % Cross-sectional Area (CSA)

                    if not( cc_value{active_slice_counter}.mm == cc_value{active_slice_counter - 1}.mm), 
                        cc_value{active_slice_counter-1}.save_length = cc_value{active_slice_counter-1}.total_length;
                        cc_value{active_slice_counter-1}.avgCSA = sumCSA / countCSA;
                        countCSA = 1;
                        sumCSA = cc_value{active_slice_counter}.area;
                    else 
                        sumCSA = sumCSA + cc_value{active_slice_counter}.area;
                        countCSA = countCSA + 1;
                    end

                    table(active_slice_counter-1,:)= [ cc_value{active_slice_counter-1}.current_slice_yz ... 
                                                       cc_value{active_slice_counter-1}.original_slice_yz ...
                                                       cc_value{active_slice_counter-1}.point ... 
                                                       cc_value{active_slice_counter-1}.circshift ... 
                                                       cc_value{active_slice_counter-1}.distance ...
                                                       cc_value{active_slice_counter-1}.int_distance_x10 ...
                                                       cc_value{active_slice_counter-1}.length_on ...
                                                       cc_value{active_slice_counter-1}.total_length ...
                                                       cc_value{active_slice_counter-1}.mm ...
                                                       cc_value{active_slice_counter-1}.save_length ...
                                                       cc_value{active_slice_counter-1}.area ...
                                                       cc_value{active_slice_counter-1}.eccent ...
                                                       cc_value{active_slice_counter-1}.majaxis ...
                                                       cc_value{active_slice_counter-1}.minaxis ...
                                                       cc_value{active_slice_counter-1}.avgCSA];

                elseif active_slice_counter == 1
                   countCSA = 1;
                   sumCSA = cc_value{active_slice_counter}.area;
                end
            end
        end
        
        if active_slice_counter == 0
            disp('no ON elements found  -  quitting');
        else
            cc_value{active_slice_counter}.save_length = cc_value{active_slice_counter}.total_length;
            cc_value{active_slice_counter}.avgCSA = sumCSA / countCSA;        
            table(active_slice_counter,:)= [ cc_value{active_slice_counter}.current_slice_yz ...  % 1                 
                                             cc_value{active_slice_counter}.original_slice_yz ... % 2               
                                             cc_value{active_slice_counter}.point ...             % 3 4   
                                             cc_value{active_slice_counter}.circshift ...         % 5 6       
                                             cc_value{active_slice_counter}.distance ...          % 7       
                                             cc_value{active_slice_counter}.int_distance_x10 ...  % 8              
                                             cc_value{active_slice_counter}.length_on ...         % 9     
                                             cc_value{active_slice_counter}.total_length ...      % 10        
                                             cc_value{active_slice_counter}.mm ...                % 11
                                             cc_value{active_slice_counter}.save_length ...       % 12           
                                             cc_value{active_slice_counter}.area ...              % 13 
                                             cc_value{active_slice_counter}.eccent ...            % 14
                                             cc_value{active_slice_counter}.majaxis ...           % 15
                                             cc_value{active_slice_counter}.minaxis ...           % 16
                                             cc_value{active_slice_counter}.avgCSA ];             % 17    

            length=table(:, 12);
            length(length==0) = [];

            careas=table(:, 17);
            careas(careas==0) = [];

            mNms=table(:, 11);
            mNms(careas==0) = [];
            
            
            subsidInd = (isbj - 1) * 2 + side;

            loopRef(subsidInd,:)=[sbj{1}, {'on'}, {sides{side}}];
            for typ=1:16
                noopRef(typ,:)=[loopRef(subsidInd,:) tablabels(1,typ)];
            end
            tablelength(subsidInd, 1:16)=[ cc_value{active_slice_counter}.total_length  ...
                                           length(1) ... 
                                           length(2)-length(1) ... 
                                           length(3)-length(2) ... 
                                           length(4)-length(3) ... 
                                           cc_value{active_slice_counter}.total_length-length(4) ... 
                                           careas(1) ... 
                                           careas(2) ... 
                                           careas(3) ... 
                                           careas(4) ... 
                                           careas(5) ... 
                                           mNms(1)   ...
                                           mNms(2)   ...
                                           mNms(3)   ...
                                           mNms(4)   ...
                                           mNms(5)];   

    %        get the names for the various subelements and insert them in a
    %        loopRef type structre for naming.
            xlRange=strcat('A',num2str((subsidInd - 1) * size(table, 2) + 1));
            xlwrite(DataFile, noopRef, 'Sheet 1', xlRange);      
            xlRange=strcat('E',num2str((subsidInd - 1) * size(table, 2) + 1));  
            xlwrite(DataFile, table', 'Sheet 1', xlRange);

            bbb = cc_value{1}.vol ; % start with first slice
            interpolation_counter = 10; % minislice counter

            n_slice=size(cc_value, 2); % count the nonzero original slices
            

            disp(strcat('INFO: Nerve Interpolation...'));

            % calculate the volumes for the remaining slices
            for ii=2:n_slice % loop over the nonzero slices
                
                % get the length in terms of minislices needed for this slice to become
                n_dist = size(cc_value{ii}.intra, 2); 
                
                % insert the the extra slices as needed
                new_interval = interpolation_counter+1 : interpolation_counter+n_dist; 
                
                bbb(1:end, new_interval, 1:end) = cc_value{ii}.intra; 
                
                % advance the minislice counter
                interpolation_counter = interpolation_counter + n_dist; 
                
                % make a range for the 10 minislices for this slice
                interval = interpolation_counter+1 : interpolation_counter+10; 
                
                % insert the 10 minislices for this slice.
                bbb(1:end, interval, 1:end) = cc_value{ii}.vol; 
                
                % advance the minislice counter
                interpolation_counter = interpolation_counter + 10;      
            end

            original_linearized_slices = interpolation_counter;

            disp(strcat('INFO: Image creation...'));
            if interpolation_counter < maxNslices
                for ii=(interpolation_counter+1):maxNslices
                    bbb(1:end,ii,1:end) = cc_value{active_slice_counter}.vol(1:end, 1, 1:end) * 0; % zerofill the rest of the maxNslices
                end
            else
                disp(strcat('Houston, we have a problems! More than', num2str(maxNslices) ,' slices!!!'))
                quit
            end

            %fill hole - if neighbouring slices not zero, copy in slice after
            hole_list = [];
            hole_counter = 0; 

            disp(strcat('INFO: Hole filling...'));

            for slice=2:(maxNslices-2)
                
                if max(max(bbb(1:end,slice+1,1:end))) == 0 && ...
                   max(max(bbb(1:end,slice,1:end))) > 0 && ...
                   max(max(bbb(1:end,slice+2,1:end))) > 0
                    hole_counter=hole_counter+1;
                    hole_list(hole_counter)=slice;
                    bbb(1:end,slice+1,1:end) = bbb(1:end,slice+2,1:end);
                end
            end

            % adapted to label each line of values rather than sheet names
            xlRange=strcat('A',num2str(subsidInd));
            xlwrite(HoleFile,loopRef(subsidInd,:),'Sheet 1', xlRange);
            if ~isempty(hole_list)
                 xlRange=strcat('D',num2str(subsidInd));
                 xlwrite(HoleFile,hole_list,'Sheet 1', xlRange);
            end

            % copy the original header and adapt it to the aligned volume (bbb) 
            dd=aa;
            dd.hdr=aa.original.hdr;
            dd.hdr.dime.pixdim(3)=dd.hdr.dime.pixdim(3)/10;
            dd.hdr.dime.dim(3)=size(bbb,2);
            dd.img=bbb;

            disp(strcat('INFO: Disk writing...'));

            save_nifti(dd,strcat(outImPath,'/',sbj{1},'/on',sides{side},Lin4image)); 
            save(strcat(outImPath,'/',sbj{1},'/on_',sides{side},Lin4mat),'cc_value');

            lengthfactor=maxNslices/round(cc_value{active_slice_counter}.total_length/dd.hdr.dime.pixdim(3)); % convert between MaxNslices and the current number of slices (after straightening)
            zz=cc_value{1}.vol(1:end,1,1:end)*0; % start with an empty slice

            check_range=zeros(maxNslices,1);

            disp(strcat('INFO: Normalization...'));

            for ii=1:maxNslices           
                jj=round(ii / maxNslices * original_linearized_slices); %figure out slice in aligned that needs to go into the ii-th slice of the normalized
                if jj<1, jj=1; end 
                check_range(ii)=jj; 
                zz(1:end,ii,1:end)=bbb(1:end,jj,1:end); % the ii-th fslice in a volume of maxNslices is the jj-th slice of the  original

                if ii > 2 % if a hole is found between slices fill it with current
                   if check_range(ii-2) > 0 && check_range(ii-1) == 0 && check_range(ii) > 0
                        zz(1:end,ii-1,1:end) = zz(1:end,ii-2,1:end);
                   end
                end
            end

            xx=dd; % use dd for the header information              
            xx.img=zz;
            save_nifti(xx,strcat(outImPath,'/',sbj{1},'/on',sides{side},Norm4image));
            save(strcat(outImPath,'/',sbj{1},'/on_',sides{side},Norm4mat),'cc_value');

            % adapted to label each line of values         
            xlRange=strcat('A',num2str(subsidInd+1));
            xlwrite(RangeFile,loopRef(subsidInd,:),'Sheet 1', xlRange);
            xlRange=strcat('D',num2str(subsidInd+1));
            xlwrite(RangeFile,check_range','Sheet 1', xlRange);
        end
    end
end

xlRange=char('A1');
xlwrite(LenStretchFile,stretxt,'Sheet 1',xlRange);
xlRange=char('A2');
xlwrite(LenStretchFile,loopRef,'Sheet 1',xlRange);
xlRange=char('D2'); 
xlwrite(LenStretchFile,tablelength,'Sheet 1',xlRange);

%ensure isotropicness...
setenv( 'FSLOUTPUTTYPE' , 'NIFTI_GZ' );
ResampleScript = fullfile(pwd,'aVP_resample.sh');
system(ResampleScript);

ResampDataFile=strcat(outResPath,'/aVP_slice_data_iso.xlsx');                     
ResampStretchFile=strcat(outResPath,'/aVP_section_CSA_length_iso.xlsx');


resamptablelength=[];
isbj=0;
resamp={'linearize','normalized' };

resamplabels=[ {'curr_sli_yz'}, {'orig_sli_yz'}, {'mMax'}, {'dist'}, {'tot_len'}, {'save_len'}, {'CSArea'}, {'Eccent'}, {'MajAxis'}, {'MinAxis'}, {'AvgCSA'} ];
resampstretxt=[ {'Subject'} {'Image'} {'ONsection'} {'side'} {'TotLength'} {'OT_length'} {'OC_length'} {'iCran_length'} {'iCan_length'} {'iOrb_length'} {'OT_CSA'} {'OC_CSA'} {'iCran_CSA'} {'iCan_CSA'} {'iOrb_CSA'} {'SegmCode 1'} {'SegmCode 2'} {'SegmCode 3'} {'SegmCode 4'} {'SegmCode 5'}];   

for sbj = subject_list{1}'
    isbj=isbj+1;
    for side=1:2 ; % :2
        for rr=1:2;
            clear cc_value
            bname=strcat(sbj{1},'/on',sides{side},'_',resamp{rr},'_4bc_iso06')
            fname=strcat(inPath,'/',bname);
            
            aa=load_nifti(fname);
            oo=aa;
        % i,j,k == x, y, z
        
        % x,z il piano di sezione ...
            x_dim=size(aa.img,1);
            z_dim=size(aa.img,3);
            dy=size(aa.img,2);
        
            x_resolution=aa.hdr.dime.pixdim(2);
            z_resolution=aa.hdr.dime.pixdim(4);
            y_resolution=aa.hdr.dime.pixdim(3);
                
            rtable = [];
            incount=0;
            countCSA = 0;
            sumCSA = 0;
            for y=1:dy  
              selected_y_slice=squeeze(aa.img(:,y,:)); 
              max_voxel_value=max(max(selected_y_slice)); 
              if max_voxel_value>0 
                incount = incount + 1;
                binarized_slice=selected_y_slice>0;               
                slice_properties = regionprops(binarized_slice,'Area','MajorAxisLength','MinorAxisLength','Eccentricity');   
                cc_value{incount}.current_slice_yz=incount;
                cc_value{incount}.original_slice_yz=y;
                cc_value{incount}.mm = max_voxel_value ;   
                cc_value{incount}.distance = y_resolution ;
                cc_value{incount}.majaxis = slice_properties(1).MajorAxisLength*x_resolution ;
                cc_value{incount}.minaxis = slice_properties(1).MinorAxisLength*z_resolution ;
                cc_value{incount}.area = slice_properties(1).Area * x_resolution * z_resolution ;
                cc_value{incount}.eccent = slice_properties(1).Eccentricity ;

                cc_value{incount}.save_length = 0;
                cc_value{incount}.avgCSA = 0;
                countCSA = countCSA + 1;
                sumCSA = sumCSA + cc_value{incount}.area ;
                
                if incount > 1
                    cc_value{incount}.total_length = cc_value{incount-1}.total_length + cc_value{incount}.distance;
                    if not( cc_value{incount}.mm == cc_value{incount - 1}.mm) 
                        cc_value{incount-1}.save_length = cc_value{incount-1}.total_length;
                        cc_value{incount-1}.avgCSA = sumCSA / countCSA;
                        countCSA = 1;
                        sumCSA = cc_value{incount}.area;
                    else 
                        sumCSA = sumCSA + cc_value{incount}.area;
                        countCSA = countCSA + 1;
                    end
                    rtable(incount-1,:)= [ double(cc_value{incount-1}.current_slice_yz) double(cc_value{incount-1}.original_slice_yz) double(cc_value{incount-1}.mm) cc_value{incount-1}.distance cc_value{incount-1}.total_length cc_value{incount-1}.save_length cc_value{incount-1}.area cc_value{incount-1}.eccent cc_value{incount-1}.majaxis cc_value{incount-1}.minaxis cc_value{incount-1}.avgCSA ];
                else 
                    cc_value{incount}.total_length = cc_value{incount}.distance ;
                end                
              end
            end
            
            if incount == 0
               disp('no ON elements found  -  quitting');
               quit
            end
            cc_value{incount}.save_length = cc_value{incount}.total_length;
            cc_value{incount}.avgCSA = sumCSA / countCSA;        
            rtable(incount,:)= [ double(cc_value{incount}.current_slice_yz) double(cc_value{incount}.original_slice_yz) double(cc_value{incount}.mm) cc_value{incount}.distance cc_value{incount}.total_length cc_value{incount}.save_length cc_value{incount}.area cc_value{incount}.eccent cc_value{incount}.majaxis cc_value{incount}.minaxis cc_value{incount}.avgCSA ];

            length=rtable(:,6);
            length(length==0)=[];
            careas=rtable(:,11);
            mNms=rtable(:,3);
            mNms(careas==0)=[];
            careas(careas==0)=[];
    
            subsidInd=(isbj-1)*4 + (rr-1)*2 + side;

            resamploopRef(subsidInd,:)=[sbj{1}, resamp{rr}, {'on'}, {sides{side}}];
            for typ=1:11
                resampnoopRef(typ,:)=[resamploopRef(subsidInd,:) resamplabels(1,typ)];
            end

    %        get the names for the various subelements and insert them in a
    %        loopRef type structre for naming.
            xlRange=strcat('A',num2str((subsidInd-1)*size(rtable,2)+1));
            xlwrite(ResampDataFile, resampnoopRef, 'Sheet 1', xlRange);      
            xlRange=strcat('F',num2str((subsidInd-1)*size(rtable,2)+1));  
            xlwrite(ResampDataFile, rtable', 'Sheet 1', xlRange);

            resamptablelength(subsidInd, 1:16)=[ cc_value{incount}.total_length  length(1) length(2)-length(1) length(3)-length(2) length(4)-length(3) cc_value{incount}.total_length-length(4) careas(1) careas(2) careas(3) careas(4) careas(5) mNms(1) mNms(2) mNms(3) mNms(4) mNms(5)];   
        end
    end
end

xlRange=char('A1');
xlwrite(ResampStretchFile,resampstretxt,'Sheet 1',xlRange);
xlRange=char('A2');
xlwrite(ResampStretchFile,resamploopRef,'Sheet 1',xlRange);
xlRange=char('E2'); 
xlwrite(ResampStretchFile,resamptablelength,'Sheet 1',xlRange);

disp('done')
