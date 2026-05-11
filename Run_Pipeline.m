restoredefaultpath;
rehash toolboxcache;

clear all
close all

packageDir = pwd;
addpath(fullfile(packageDir, 'mirt_csp_simplify'), '-begin');
addpath(genpath(fullfile(packageDir, 'Helpers')), '-begin'); 
%% Generate the synthetic MSK dynamic image using our matlab software

[motion_vol_flexion, masks_flexion, params_flexion] = generateMSKNonRigidMotion_lep(...
    'VolSize', [128, 128, 128], ...
    'NumMotionStates', 10, ...
    'MotionType', 'flexion', ...
    'FlexionAngle', [-30, 20], ...
    'AnatomicalModel', 'complex', ...
    'IncludeCartilage', true, ...
    'IncludeLigaments', true, ...
    'RandomSeed', 42);

%% Generate the golden angle 3D radial trajectory using our matlab software
baseResolution = 128;
nSpokes = 44800;
fovx_mm = 210;
osFactor = 2;

fprintf('Parameters:\n');
fprintf('  baseResolution: %d\n', baseResolution);
fprintf('  nSpokes: %d\n', nSpokes);
fprintf('  fovx_mm: %.2f\n', fovx_mm);
fprintf('  osFactor: %d\n', osFactor);

% Generate trajectory
[traj_mm, polarAngle, azimuthalAngle,traj_scaled] = ...
    generateGoldenAngleTrajectory_lep(baseResolution, nSpokes, fovx_mm, osFactor, 'siemens');

fprintf('\nGenerated trajectory dimensions:\n');
fprintf('  traj  shape: [%d, %d, %d]\n', size(traj_mm, 1), size(traj_mm, 2), size(traj_mm, 3)); 
fprintf('  polarAngle shape: [%d]\n', length(polarAngle));
fprintf('  azimuthalAngle shape: [%d]\n', length(azimuthalAngle));
%% Generate the synthetic coil sensitivity profile using mirt matlab software
NumCoil = 18;
nx = 128;
ny = nx;
nz = nx;

CSP = ir_mri_sensemap_sim('nx', nx, 'ny',ny, 'nz',nz, ...
		'scale', 'ssos_center', ...
		'ncoil', NumCoil, 'orbit_start', -90, 'rcoil', 250);
fa = sqrt(sum(abs(CSP).^2,4)) ;
CSP = double(CSP./(fa+eps));  
CSP = abs(CSP);
CSP(isnan(CSP)) = 0;
% imxshow_lep_v2(CSP,'name','Synthetic CSP')
%% Use synthetic dynmic MSK images to generate multi-circle motion kSpace data

CircleNum = 10;
ImageCombined = motion_vol_flexion;
MotionNum = size(ImageCombined,4);
subset = cell(MotionNum,1);

MotionD = ceil(nSpokes/(CircleNum*MotionNum));

for it = 1:MotionNum
  inx = it:MotionNum:it+MotionNum*(CircleNum-1);
  inxs = [];
  for it1 =1:length(inx)
    inxs = [inxs,   (inx(it1)-1)*MotionD+1 : inx(it1)*MotionD  ];  
  end
  inxs(inxs>nSpokes)=[];
  subset{it} = inxs;
end

[kSpace_Mo,traj_Mo ]= Image2NonCartesiankSpace_lep(ImageCombined,CSP,traj_scaled,'subset',subset );



%% Parameter Configuration
WinL = 448*10 ;
Stride = 448*10;
[Subsets_SE] = GenerateSubsets_SliWin_lep(WinL,Stride,nSpokes);


%%  Show SpE PC
WinLen_SpE = 448;
PC = 1:4;
[SE_score] = Extra_SpE_PosInf_lep(kSpace_Mo,WinLen_SpE,PC);

 
figure('WindowStyle','normal','Color','white');
DisplayStride = 2; % To improve rendering 
colors = lines(length(PC)); 
for it = 1:length(PC)
    subplot(2,2,it)   
    xx = Scale_lep( SE_score(:,it), [0,1] );
    plot(1:DisplayStride:length(xx), xx(1:DisplayStride:end),'LineWidth',2,'Color', colors(it,:))
   
    EnhancePlot_lep; 
    title(['SpE PC',num2str(it)])
    xlim([1,length(xx)]),ylim([-0.1,1.1])
end

%%  Non-Reordering reconstruction
[kSpace_CC,sv] = CompressedCoil_lep(kSpace_Mo,'r',4); 
img_Non_Reordering = tf_nufft_radial_recon_lep(kSpace_CC, traj_Mo, baseResolution, fovx_mm, 'down_sampling_factor', 1,'subsets',Subsets_SE);  
clear kSpace_CC
% imxshow_lep_v2(img_Non_Reordering,'plane','c' )
%%  the proposed framework for reordering dynamic reconstruction
 
SpE_Rec_PC = cell(4,1);
for PC =1:4
    % ===============  spoke-energy extraction (with different PCA component selection) ==================
    [SE_score] = Extra_SpE_PosInf_lep(kSpace_Mo,WinLen_SpE,PC);
 
    %   ============== motion detection and Reordering ============== 
    [kSpace_SEOrd,traj_SEOrd,SE_score_Order] = Reordering_Spoke_lep(kSpace_Mo, traj_Mo, SE_score, WinLen_SpE);
 
    %   ==============  reconstruction/correction output ============== 
    [kSpace_CC,sv] = CompressedCoil_lep(kSpace_SEOrd,'r',4); 
    img_SpE_Reordering = tf_nufft_radial_recon_lep(kSpace_CC, traj_SEOrd, baseResolution, fovx_mm, 'down_sampling_factor', 1,'subsets',Subsets_SE);  
    clear kSpace_CC
    SpE_Rec_PC{PC} = img_SpE_Reordering;

end

% imxshow_lep_v2(SpE_Rec_PC,'name','PC','plane','c')
%% %%%%%%%%%%%%%%%%%%%%%%%%% quantitive evaluation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

ResultCell = cell(6,2);

SliceInd = 50:90;

% -------- GT --------

[se_m, se_s] = Quantitve_Cal_lep(@spectral_entropy_lep, ImageCombined, SliceInd, []);
[ave_m, ave_s] = Quantitve_Cal_lep(@average_edge_strength_lep, ImageCombined, SliceInd, []);

ResultCell{1,1} = sprintf('%.2f ± %.2f', se_m, se_s);
ResultCell{1,2} = sprintf('%.2f ± %.2f', ave_m, ave_s);

% -------- No Reordering --------

[se_m, se_s] = Quantitve_Cal_lep(@spectral_entropy_lep, img_Non_Reordering, SliceInd, []);
[ave_m, ave_s] = Quantitve_Cal_lep(@average_edge_strength_lep, img_Non_Reordering, SliceInd, []);

ResultCell{2,1} = sprintf('%.2f ± %.2f', se_m, se_s);
ResultCell{2,2} = sprintf('%.2f ± %.2f', ave_m, ave_s);

% -------- SpE PC1-PC4 --------

for it = 1:4

    row = it + 2;

    [m, s] = Quantitve_Cal_lep(@spectral_entropy_lep, SpE_Rec_PC{it}, SliceInd, []);
    ResultCell{row,1} = sprintf('%.2f ± %.2f', m, s);

    [m, s] = Quantitve_Cal_lep(@average_edge_strength_lep, SpE_Rec_PC{it}, SliceInd, []);
    ResultCell{row,2} = sprintf('%.2f ± %.2f', m, s);

end

% -------- Table --------

VarNames = {'SE','AES'};

RowNames = {'GT','NoReord','SpE_PC1','SpE_PC2','SpE_PC3','SpE_PC4'};

T = cell2table(ResultCell,'VariableNames',VarNames,'RowNames',RowNames);

disp(T)



%%   Show all the results for comparison

SliceIdx = 64;

MethodNames = {'GT','NReord','PC1','PC2','PC3','PC4'};

SpE_Rec_PC1 = ChangeView_lep(SpE_Rec_PC,'c');
SpE_Rec_PC1 = cellfun(@(x) flip(x,1),SpE_Rec_PC1,'UniformOutput',false);

img_Non_Reordering1 = ChangeView_lep(img_Non_Reordering,'c');
img_Non_Reordering1 = flip(img_Non_Reordering1,1);

ImageCombined1 = ChangeView_lep(ImageCombined,'c');
ImageCombined1 = flip(ImageCombined1,1);
ImageCombined1 = flip(ImageCombined1,4);


MakeLongImage = @(x) reshape(permute(squeeze(x),[1 3 2]),size(x,1)*size(x,4),size(x,2));

A0 = MakeLongImage(ImageCombined1(:,:,SliceIdx,:));
A1 = MakeLongImage(img_Non_Reordering1(:,:,SliceIdx,:));
A2 = MakeLongImage(SpE_Rec_PC1{1}(:,:,SliceIdx,:));
A3 = MakeLongImage(SpE_Rec_PC1{2}(:,:,SliceIdx,:));
A4 = MakeLongImage(SpE_Rec_PC1{3}(:,:,SliceIdx,:));
A5 = MakeLongImage(SpE_Rec_PC1{4}(:,:,SliceIdx,:));

GapPix = 6;
Gap = zeros(size(A0,1),GapPix);

ALL = [A0 Gap A1 Gap A2 Gap A3 Gap A4 Gap A5];

fig = figure('WindowStyle','normal','Color','black');

imshow(ALL,[])

axis image
axis off
colormap(gray)

drawnow

ImgH = size(ALL,1);
ImgW = size(ALL,2);

ScaleFactor = 0.65;

set(fig,'Units','pixels','Position',[100 50 ImgW*ScaleFactor ImgH*ScaleFactor]);

ax = gca;
ax.Position = [0.08 0.02 0.90 0.96];

hold on

SingleWidth = size(A0,2) + GapPix;

for k = 1:length(MethodNames)

    Xcenter = (k-1)*SingleWidth + size(A0,2)/2;

    text(Xcenter,-15,MethodNames{k},'Color',[1 1 1],'FontSize',13,'FontWeight','bold','HorizontalAlignment','center');

end

annotation(fig,'textarrow',[0.055 0.055],[0.68 0.42],'String','T','Color',[1 1 1],'FontSize',13,'LineWidth',2,'TextRotation',0,'Interpreter','none');

%%
ShowMetrics = 1;

if ShowMetrics

    TextY1 = size(ALL,1) + 35;
    TextY2 = size(ALL,1) + 60;

    for k = 1:length(MethodNames)

        Xcenter = (k-1)*SingleWidth + size(A0,2)/2;

        SE_text = split(T.SE{k},' ± ');
        SE_text = SE_text{1};

        AES_text = split(T.AES{k},' ± ');
        AES_text = AES_text{1};

        text(Xcenter,TextY1,['SE: ' SE_text], ...
            'Color',[1 1 1], ...
            'FontSize',10, ...
            'FontWeight','bold', ...
            'HorizontalAlignment','center');

        text(Xcenter,TextY2,['AES: ' AES_text], ...
            'Color',[1 1 1], ...
            'FontSize',10, ...
            'FontWeight','bold', ...
            'HorizontalAlignment','center');

    end

    ax.Position = [0.08 0.10 0.90 0.88];

end