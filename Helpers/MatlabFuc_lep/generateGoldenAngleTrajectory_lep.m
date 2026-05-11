function [traj_mm, polarAngle, azimuthalAngle,trajRAD] = generateGoldenAngleTrajectory_lep(baseResolution, nSpokes, fovx_mm, osFactor, set_view)
% generateGoldenAngleTrajectory_lep - Generate 3D golden angle radial k-space trajectory
%
% 100% EXACT COPY of siemens_data_loader.m trajectory generation pipeline
% Mimics: calculateRadialTrajGolden3D_modifiedByFatih.m + coordinate transformations
%
% SYNTAX:
%   [traj_mm, polarAngle, azimuthalAngle, trajRAD] = ...
%       generateGoldenAngleTrajectory_lep(baseResolution, nSpokes, fovx_mm, osFactor, set_view)
%
% INPUT:
%   baseResolution  - Base resolution of k-space (samples per spoke)
%   nSpokes         - Number of spokes
%   fovx_mm         - Field of view in mm
%   osFactor        - Oversampling factor (default: 1)
%   set_view        - Coordinate system: 'siemens', 'axial', 'sagittal', 'coronal'
%
% OUTPUT:
%   traj_mm         - Trajectory [nCol, nSpokes, 3] with coordinate transformation
%   polarAngle      - Polar angle for each spoke
%   azimuthalAngle  - Azimuthal angle for each spoke
%   trajRAD         - Normalized trajectory [-0.5, 0.5] (for reference)
%
% AUTHOR: Enping Lin, PhD. 
% DATE: 2026.5.6

    % Default parameters
    if nargin < 4
        osFactor = 1;
    end
    if nargin < 5
        set_view = 'siemens';
    end
    
    % ========================================================================
    % From calculateRadialTrajGolden3D_modifiedByFatih.m
    % ========================================================================
    
    params.baseResolution = baseResolution;
    params.nSpokes = nSpokes;
    params.fovx_mm = fovx_mm;
    params.osFactor = osFactor;
    
    % Compute golden means ratios in 3D
    Mfib3d = [0, 1, 0; 0, 0, 1; 1, 0, 1];
    [V3d, ~] = eig(Mfib3d);
    v = V3d(:,1)/V3d(3,1);
    m_phi1 = round(v(1), 15);
    m_phi2 = round(v(2), 15);
    
    baseresolution = params.baseResolution;
    nSpokes = params.nSpokes;
    fovx_mm = params.fovx_mm;
    osFactor = params.osFactor;
    nCol = baseresolution*osFactor;
    
    maxkspace = 0.5*baseresolution/fovx_mm;
    inc = 1.0/(osFactor*fovx_mm);
    rho = (-maxkspace(1):inc(1):maxkspace(1)-inc(1))';
    
    nSpokesRemove = 0;
    
    % kspace trajectory
    trajx = zeros(nCol,nSpokes);
    trajy = zeros(nCol,nSpokes);
    trajz = zeros(nCol,nSpokes);
    
    polarAngle = zeros(1,nSpokes);
    azimuthalAngle = zeros(1,nSpokes);
    m1 = zeros(1,nSpokes);
    m2 = zeros(1,nSpokes);
    
    polarAngle(1) = pi/2;
    trajx(:,1) = rho;
    
    for ii=2:nSpokes
        m1(ii) = mod( (m1(ii-1) + m_phi1), 1);
        m2(ii) = mod( (m2(ii-1) + m_phi2), 1);
        
        polarAngle(ii) = pi/2 + m1(ii)*pi/2;
        azimuthalAngle(ii) = 2*pi*m2(ii);
        
        xA = cos(azimuthalAngle(ii))*sin(polarAngle(ii));
        yA = sin(azimuthalAngle(ii))*sin(polarAngle(ii));
        zA = cos(polarAngle(ii));
        len = sqrt(xA^2+yA^2+zA^2);
        
        trajx(:,ii) = rho*(xA/len);
        trajy(:,ii) = rho*(yA/len);
        trajz(:,ii) = rho*(zA/len);
    end
    
    trajxRAD = reshape(trajx(:,nSpokesRemove+1:end),[],1);
    trajyRAD = reshape(trajy(:,nSpokesRemove+1:end),[],1);
    trajzRAD = reshape(trajz(:,nSpokesRemove+1:end),[],1);
    
    trajRAD(:,1) = trajxRAD(:);
    trajRAD(:,2) = trajyRAD(:);
    trajRAD(:,3) = trajzRAD(:);
    
    trajRAD_mm = trajRAD;
    trajRAD = trajRAD/(2*max(max(abs(trajRAD))));
    trajRAD = reshape(trajRAD , [nCol, params.nSpokes, 3]);
    % ========================================================================
    % From siemens_data_loader.m - Reshape and coordinate transformation
    % ========================================================================
    
    traj_mm = trajRAD_mm;  % Use UNSCALED trajectory (trajRAD_mm, not trajRAD)
    traj_mm = reshape(traj_mm, [nCol, params.nSpokes, 3]);
    
    % Define Permutation matrix (from siemens_data_loader)
    Permutation = [0 1 0; 1 0 0; 0 0 1];
    
    % Define rotation matrix based on set_view (EXACT copy from siemens_data_loader)
    rotation_matrix = [];
    switch set_view
        case 'siemens'
            rotation_matrix = eye(3);
        case 'sagittal'
            rotation_matrix = [0 1 0; -1 0 0; 0 0 1]*Permutation;
        case 'axial'
            rotation_matrix = [-1 0 0; 0 0 -1; 0 -1 0]*Permutation;
        case 'coronal'
            rotation_matrix = [0 1 0; 0 0 1; 1 0 0]*Permutation;
        otherwise
            error('set_view must be one of: siemens, sagittal, axial, coronal');
    end
    
    % Apply rotation transformation (EXACT copy from siemens_data_loader)
    if ~isempty(rotation_matrix)
        rotation_matrix = round(rotation_matrix);
        traj_mm = reshape(reshape(traj_mm, [], 3)*rotation_matrix', size(traj_mm));
    end

end
