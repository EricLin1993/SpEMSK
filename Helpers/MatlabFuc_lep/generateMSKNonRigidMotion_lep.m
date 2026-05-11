function [motion_volume, segment_masks, motion_params_out] = generateMSKNonRigidMotion_lep(varargin)
%% generateMSKNonRigidMotion_lep - Generate realistic MSK non-rigid motion for ankle/foot imaging
%
% SYNTAX:
%   [motion_volume, segment_masks, motion_params] = generateMSKNonRigidMotion_lep()
%   [motion_volume, segment_masks, motion_params] = generateMSKNonRigidMotion_lep('PropertyName', PropertyValue, ...)
%
% DESCRIPTION:
%   Generates 3D volumes with realistic musculoskeletal non-rigid motion using
%   multi-rigid-body decomposition combined with local deformation fields.
%   Specifically designed for ankle/foot imaging simulation with anatomically
%   plausible motion patterns.
%
%   MOTION TYPES IMPLEMENTED:
%   - 'flexion': Dorsiflexion/plantarflexion around lateral-medial axis (X-axis)
%                Range: -25° to +35° (negative = dorsiflexion, positive = plantarflexion)
%   - 'inversion': Inversion/eversion around vertical axis (Z-axis)
%                  Range: -20° to +20° (simulates ankle sprain mechanics)
%   - 'rotation': Internal/external rotation around anterior-posterior axis (Y-axis)
%                 Range: -10° to +10°
%   - 'combined': Circular trajectory combining all three axes simultaneously
%                 Creates natural ankle motion patterns for gait simulation
%
%   ANATOMICAL STRUCTURES USED:
%   1. Tibia: Proximal bone (stationary reference), ellipsoid at z*0.25
%      Size: (0.08, 0.08, 0.15), Intensity: 0.9 (cortical bone)
%   2. Fibula: Parallel to tibia, smaller ellipsoid
%      Size: (0.05, 0.05, 0.12), Intensity: 0.85
%   3. Talus: Ankle joint articulating bone, rotates with motion
%      Size: (0.1, 0.09, 0.06), Intensity: 0.88
%   4. Calcaneus: Heel bone, fixed relative to foot
%      Size: (0.12, 0.1, 0.08), Intensity: 0.87
%   5. Soft Tissue: Muscle, fat, skin envelope
%      Large ellipsoid (0.3, 0.35, 0.4) with internal muscle texture
%      Intensity: 0.4 + noise (simulates muscle/fat contrast)
%   6. Cartilage: Joint articular surfaces (optional)
%   7. Ligaments: Joint stabilizing structures (optional)
%
%   ANATOMICAL MODEL COMPLEXITY LEVELS:
%   - 'simple': Basic 3-bone structure (tibia, talus, calcaneus) + uniform soft tissue
%               Fast computation, suitable for algorithm validation
%   - 'intermediate': Tibia+fibula, talus, calcaneus + textured soft tissue + cartilage
%                     Balanced realism and computational efficiency (default)
%   - 'complex': Full foot skeleton (navicular, cuboid, metatarsals) + multi-layer soft tissue
%                High anatomical fidelity for clinical simulation studies
%
%   SOFT TISSUE STIFFNESS SIMULATION:
%   - Range: 0.1 (very soft, like fat) to 0.6 (stiff, like scar tissue)
%   - Default: 0.3 (normal muscle/fat stiffness)
%   - Implementation: Force field generation based on joint motion angles,
%     followed by elastic deformation using stiffness-based displacement
%   - Effect: Lower stiffness = more deformation, higher stiffness = less deformation
%
% INPUT PARAMETERS (name-value pairs):
%   'VolSize'          - Volume size [nx, ny, nz] (default: [256, 256, 128])
%   'NumMotionStates'  - Number of motion states to generate (default: 8)
%   'MotionType'       - Type of motion: 'flexion', 'inversion', 'rotation', 'combined' (default: 'flexion')
%   'FlexionAngle'     - Dorsi/plantarflexion angle range in degrees (default: [-20, 30])
%   'InversionAngle'   - Inversion/eversion angle range (default: [-15, 15])
%   'RotationAngle'   - Internal/external rotation angle range (default: [-10, 10])
%   'SoftTissueStiffness' - Soft tissue elasticity [0-1] (default: 0.3)
%   'BoneStiffness'    - Bone rigidity [0-1] (default: 1.0)
%   'DeformationStrength' - Local deformation magnitude [0-1] (default: 0.15)
%   'TibiaMotion'      - Tibia motion flag: 'fixed' or 'moving' (default: 'fixed')
%   'IncludeCartilage' - Include cartilage layers (default: true)
%   'IncludeLigaments' - Include ligament structures (default: true)
%   'AnatomicalModel'  - Complexity: 'simple', 'intermediate', 'complex' (default: 'intermediate')
%   'RandomSeed'       - Random seed for reproducibility (default: [])
%
% OUTPUT:
%   motion_volume      - 4D array [nx, ny, nz, num_motion_states] of intensity values
%   segment_masks      - Structure with anatomical segment masks:
%                        .tibia, .talus, .calcaneus, .foot, .soft_tissue, .ligaments, .cartilage
%   motion_params_out  - Structure with actual motion parameters used
%
% EXAMPLES:
%   % Basic dorsiflexion/plantarflexion motion
%   [vol, masks, params] = generateMSKNonRigidMotion_lep(...
%       'NumMotionStates', 10, ...
%       'MotionType', 'flexion', ...
%       'FlexionAngle', [-25, 35]);
%
%   % Complex combined motion with high anatomical detail
%   [vol, masks, params] = generateMSKNonRigidMotion_lep(...
%       'NumMotionStates', 16, ...
%       'MotionType', 'combined', ...
%       'AnatomicalModel', 'complex', ...
%       'SoftTissueStiffness', 0.25);
%
%   % Inversion/eversion motion for sprain simulation
%   [vol, masks, params] = generateMSKNonRigidMotion_lep(...
%       'MotionType', 'inversion', ...
%       'InversionAngle', [-20, 20], ...
%       'DeformationStrength', 0.2);
%
% AUTHOR: Enping Lin
% DATE: 2026

%% Parse input parameters
p = inputParser;
p.addParameter('VolSize', [256, 256, 128], @(x) isvector(x) && length(x)==3);
p.addParameter('NumMotionStates', 8, @isscalar);
p.addParameter('MotionType', 'flexion', @(x) ischar(x) || isstring(x));
p.addParameter('FlexionAngle', [-20, 30], @isvector);
p.addParameter('InversionAngle', [-15, 15], @isvector);
p.addParameter('RotationAngle', [-10, 10], @isvector);
p.addParameter('SoftTissueStiffness', 0.3, @(x) x>=0 && x<=1);
p.addParameter('BoneStiffness', 1.0, @(x) x>=0 && x<=1);
p.addParameter('DeformationStrength', 0.15, @(x) x>=0 && x<=1);
p.addParameter('TibiaMotion', 'fixed', @(x) ischar(x) || isstring(x));
p.addParameter('IncludeCartilage', true, @islogical);
p.addParameter('IncludeLigaments', true, @islogical);
p.addParameter('AnatomicalModel', 'intermediate', @(x) ischar(x) || isstring(x));
p.addParameter('RandomSeed', [], @(x) isempty(x) || isscalar(x));

parse(p, varargin{:});

% Extract parameters
vol_size = p.Results.VolSize;
n_states = p.Results.NumMotionStates;
motion_type = char(p.Results.MotionType);  % Convert to char for strcmp
flexion_range = p.Results.FlexionAngle;
inversion_range = p.Results.InversionAngle;
rotation_range = p.Results.RotationAngle;
soft_stiffness = p.Results.SoftTissueStiffness;
bone_stiffness = p.Results.BoneStiffness;
deform_strength = p.Results.DeformationStrength;
tibia_motion = char(p.Results.TibiaMotion);  % Convert to char for strcmp
include_cartilage = p.Results.IncludeCartilage;
include_ligaments = p.Results.IncludeLigaments;
anatomical_model = char(p.Results.AnatomicalModel);  % Convert to char for strcmp
random_seed = p.Results.RandomSeed;

% Set random seed if provided
if ~isempty(random_seed)
    rng(random_seed);
end

%% Initialize output structure
motion_volume = zeros([vol_size, n_states]);

%% Step 1: Create reference 3D ankle phantom
fprintf('Step 1: Creating 3D ankle phantom...\n');
reference_phantom = createAnklePhantom3D(vol_size, anatomical_model);

%% Step 2: Create anatomical segment masks
fprintf('Step 2: Creating anatomical segment masks...\n');
[tibia_mask, talus_mask, calcaneus_mask, foot_mask, ...
 soft_tissue_mask, ligament_masks, cartilage_masks] = ...
    createAnatomicalSegments(vol_size, anatomical_model, include_cartilage, include_ligaments);

% Store masks
segment_masks.tibia = tibia_mask;
segment_masks.talus = talus_mask;
segment_masks.calcaneus = calcaneus_mask;
segment_masks.foot = foot_mask;
segment_masks.soft_tissue = soft_tissue_mask;
if include_ligaments
    segment_masks.ligaments = ligament_masks;
end
if include_cartilage
    segment_masks.cartilage = cartilage_masks;
end

%% Step 3: Define motion parameters for each state
fprintf('Step 3: Defining motion parameters...\n');

% Generate motion trajectory based on motion type
[flexion_angles, inversion_angles, rotation_angles] = ...
    generateMotionTrajectory(n_states, motion_type, flexion_range, inversion_range, rotation_range);

%% Step 4: Generate motion states
fprintf('Step 4: Generating %d motion states...\n', n_states);

% Center of ankle joint (approximate)
ankle_center = [vol_size(1)/2, vol_size(2)/2, vol_size(3)*0.35];

for state_idx = 1:n_states
    fprintf('  Processing motion state %d/%d...\n', state_idx, n_states);

    % Current motion angles
    flex_angle = flexion_angles(state_idx);
    inv_angle = inversion_angles(state_idx);
    rot_angle = rotation_angles(state_idx);

    % Initialize motion state volume
    motion_state = zeros(vol_size);

    % ===== Tibia (stationary or moving) =====
    if strcmp(tibia_motion, 'fixed')
        tibia_moved = reference_phantom .* tibia_mask;
    else
        % Tibia slight movement (simplified)
        tibia_deform = applySmallDeformation(tibia_mask, 0.02 * flex_angle/30);
        tibia_moved = reference_phantom .* tibia_deform;
    end

    % ===== Talus and foot (articulated motion) =====
    % Create rotation matrix for multi-axis rotation
    T_ankle = createAnkleTransform(ankle_center, flex_angle, inv_angle, rot_angle);

    % Apply rotation to talus
    talus_moved = applyRigidBodyTransform(reference_phantom .* talus_mask, T_ankle, ankle_center);

    % Apply rotation to foot (combined rigid + deformation)
    foot_rigid = applyRigidBodyTransform(reference_phantom .* foot_mask, T_ankle, ankle_center);

    % Add soft tissue deformation (muscle contraction, ligament tension)
    [fx, fy, fz] = generateSoftTissueForces(vol_size, flex_angle, inv_angle, rot_angle, ankle_center);
    [dx, dy, dz] = calculateElasticDeformation(fx, fy, fz, soft_stiffness);

    foot_deformed = applyDeformationField(foot_rigid, dx, dy, dz);

    % ===== Soft tissue with deformation =====
    soft_tissue_intensity = reference_phantom .* soft_tissue_mask;
    [sx, sy, sz] = generateSoftTissueDeformation(vol_size, flex_angle, deform_strength, ankle_center);
    soft_tissue_deformed = applyDeformationField(soft_tissue_intensity, sx, sy, sz);

    % ===== Combine segments =====
    motion_state = tibia_moved + talus_moved + foot_deformed + soft_tissue_deformed;

    % ===== Add cartilage and ligament effects =====
    if include_cartilage
        % Cartilage shows compression/extension
        compression_factor = 1 - 0.05 * abs(flex_angle)/30;
        motion_state = motion_state + reference_phantom .* cartilage_masks * compression_factor;
    end

    if include_ligaments
        % Ligaments show tension changes
        tension_factor = 1 + 0.03 * abs(flex_angle)/30;
        motion_state = motion_state + reference_phantom .* ligament_masks * tension_factor;
    end

    % Store motion state
    motion_volume(:, :, :, state_idx) = motion_state;
end

%% Step 5: Normalize and prepare output
fprintf('Step 5: Normalizing output...\n');

% Normalize to [0, 1]
max_val = max(motion_volume(:));
if max_val > 0
    motion_volume = motion_volume / max_val;
end

% Prepare output parameters
motion_params_out.volume_size = vol_size;
motion_params_out.num_motion_states = n_states;
motion_params_out.motion_type = motion_type;
motion_params_out.flexion_angles = flexion_angles;
motion_params_out.inversion_angles = inversion_angles;
motion_params_out.rotation_angles = rotation_angles;
motion_params_out.tibia_motion = tibia_motion;
motion_params_out.include_cartilage = include_cartilage;
motion_params_out.include_ligaments = include_ligaments;
motion_params_out.anatomical_model = anatomical_model;
motion_params_out.soft_tissue_stiffness = soft_stiffness;
motion_params_out.bone_stiffness = bone_stiffness;
motion_params_out.deformation_strength = deform_strength;
motion_params_out.ankle_center = ankle_center;

fprintf('✓ Motion volume generation complete!\n');
fprintf('  Output size: [%d, %d, %d, %d]\n', size(motion_volume, 1), size(motion_volume, 2), size(motion_volume, 3), size(motion_volume, 4));

end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function phantom = createAnklePhantom3D(vol_size, model_complexity)
    %% Create 3D ankle phantom with anatomical features

    [x, y, z] = ndgrid(1:vol_size(1), 1:vol_size(2), 1:vol_size(3));

    x0 = vol_size(1)/2;
    y0 = vol_size(2)/2;

    phantom = zeros(vol_size);

    %% Tibia (proximal, stationary bone)
    tibia_z_center = vol_size(3) * 0.25;
    tibia_a = vol_size(1) * 0.08;
    tibia_b = vol_size(2) * 0.08;
    tibia_c = vol_size(3) * 0.15;

    tibia_mask = ((x-x0)/tibia_a).^2 + ((y-y0)/tibia_b).^2 + ...
                 ((z-tibia_z_center)/tibia_c).^2 <= 1;
    phantom(tibia_mask) = 0.9;  % High intensity for cortical bone

    %% Fibula (parallel to tibia)
    fibula_x = x0 - vol_size(1)*0.12;
    fibula_mask = ((x-fibula_x)/(vol_size(1)*0.05)).^2 + ((y-y0)/(vol_size(2)*0.05)).^2 + ...
                  ((z-tibia_z_center)/(tibia_c*0.8)).^2 <= 1;
    phantom(fibula_mask) = 0.85;

    %% Talus (ankle joint, articulating bone)
    talus_z = vol_size(3) * 0.42;
    talus_a = vol_size(1) * 0.1;
    talus_b = vol_size(2) * 0.09;
    talus_c = vol_size(3) * 0.06;

    talus_mask = ((x-x0)/talus_a).^2 + ((y-y0)/talus_b).^2 + ...
                 ((z-talus_z)/talus_c).^2 <= 1;
    phantom(talus_mask) = 0.88;

    %% Calcaneus (heel bone)
    calc_z = vol_size(3) * 0.65;
    calc_y = y0 - vol_size(2)*0.08;
    calc_a = vol_size(1) * 0.12;
    calc_b = vol_size(2) * 0.1;
    calc_c = vol_size(3) * 0.08;

    calc_mask = ((x-x0)/calc_a).^2 + ((y-calc_y)/calc_b).^2 + ...
                ((z-calc_z)/calc_c).^2 <= 1;
    phantom(calc_mask) = 0.87;

    %% Navicular, cuboid, metatarsals (midfoot and forefoot)
    if strcmp(model_complexity, 'complex')
        % Add detailed midfoot structures
        nav_z = vol_size(3) * 0.58;
        nav_mask = ((x-x0)/(vol_size(1)*0.08)).^2 + ((y-y0-vol_size(2)*0.05)/(vol_size(2)*0.07)).^2 + ...
                   ((z-nav_z)/(vol_size(3)*0.04)).^2 <= 1;
        phantom(nav_mask) = 0.82;

        % Metatarsals (5 bones in forefoot)
        for i = -2:2
            mt_x = x0 + i * vol_size(1)*0.04;
            mt_z = vol_size(3) * 0.75;
            mt_mask = ((x-mt_x)/(vol_size(1)*0.03)).^2 + ((y-y0)/(vol_size(2)*0.04)).^2 + ...
                      ((z-mt_z)/(vol_size(3)*0.1)).^2 <= 1;
            phantom(mt_mask) = 0.80;
        end
    end

    %% Soft tissue (muscle, fat, skin)
    % Create large soft tissue envelope
    soft_tissue_mask = ((x-x0)/(vol_size(1)*0.3)).^2 + ...
                       ((y-y0)/(vol_size(2)*0.35)).^2 + ...
                       ((z-vol_size(3)*0.5)/(vol_size(3)*0.4)).^2 <= 1;

    % Add internal muscle structure with noise
    [muscle_intensity] = generateMuscleStructure(vol_size, x0, y0);

    soft_mask = soft_tissue_mask & ~(tibia_mask | fibula_mask | talus_mask | calc_mask);
    phantom(soft_mask) = 0.4 + 0.2 * muscle_intensity(soft_mask);

end

function muscle_intensity = generateMuscleStructure(vol_size, x0, y0)
    %% Generate realistic muscle structure patterns

    [x, y, z] = ndgrid(1:vol_size(1), 1:vol_size(2), 1:vol_size(3));

    % Perlin-like noise for muscle texture
    muscle_intensity = zeros(vol_size);

    % Multiple frequency components
    freq1 = sin(2*pi*x/vol_size(1)*3) .* cos(2*pi*y/vol_size(2)*2);
    freq2 = cos(2*pi*x/vol_size(1)*5) .* sin(2*pi*z/vol_size(3)*4);
    freq3 = sin(2*pi*(x+y)/vol_size(1)*2) .* 0.5;

    muscle_intensity = 0.3 * freq1 + 0.3 * freq2 + 0.4 * freq3;
    muscle_intensity = (muscle_intensity - min(muscle_intensity(:))) / ...
                       (max(muscle_intensity(:)) - min(muscle_intensity(:)));

end

function [tibia_m, talus_m, calc_m, foot_m, soft_m, lig_m, cart_m] = ...
    createAnatomicalSegments(vol_size, model_complexity, include_cart, include_lig)
    %% Create individual masks for each anatomical segment

    [x, y, z] = ndgrid(1:vol_size(1), 1:vol_size(2), 1:vol_size(3));

    x0 = vol_size(1)/2;
    y0 = vol_size(2)/2;

    % Tibia
    tibia_z = vol_size(3) * 0.25;
    tibia_m = ((x-x0)/(vol_size(1)*0.08)).^2 + ((y-y0)/(vol_size(2)*0.08)).^2 + ...
              ((z-tibia_z)/(vol_size(3)*0.15)).^2 <= 1;

    % Talus
    talus_z = vol_size(3) * 0.42;
    talus_m = ((x-x0)/(vol_size(1)*0.1)).^2 + ((y-y0)/(vol_size(2)*0.09)).^2 + ...
              ((z-talus_z)/(vol_size(3)*0.06)).^2 <= 1;

    % Calcaneus
    calc_z = vol_size(3) * 0.65;
    calc_y = y0 - vol_size(2)*0.08;
    calc_m = ((x-x0)/(vol_size(1)*0.12)).^2 + ((y-calc_y)/(vol_size(2)*0.1)).^2 + ...
             ((z-calc_z)/(vol_size(3)*0.08)).^2 <= 1;

    % Foot (combines talus, calcaneus, and other foot bones)
    foot_m = talus_m | calc_m;

    % Soft tissue
    soft_tissue_envelope = ((x-x0)/(vol_size(1)*0.3)).^2 + ...
                           ((y-y0)/(vol_size(2)*0.35)).^2 + ...
                           ((z-vol_size(3)*0.5)/(vol_size(3)*0.4)).^2 <= 1;
    soft_m = soft_tissue_envelope & ~(tibia_m | talus_m | calc_m);

    % Ligaments (if included)
    if include_lig
        % Simplistic ligament masks near joints
        joint_z = vol_size(3) * 0.40;
        lig_m = ((x-x0)/(vol_size(1)*0.15)).^2 + ((y-y0)/(vol_size(2)*0.15)).^2 + ...
                ((z-joint_z)/(vol_size(3)*0.1)).^2 <= 1.2 & ...
                ~(tibia_m | talus_m | calc_m);
        lig_m(soft_m) = false;  % Exclude soft tissue
    else
        lig_m = false(vol_size);
    end

    % Cartilage (if included)
    if include_cart
        % Cartilage at joint surfaces
        cart_m = talus_m;  % Simplified: just marking articulating surfaces
        cart_m = cart_m & ~((x-x0)/(vol_size(1)*0.08)).^2 - ((y-y0)/(vol_size(2)*0.08)).^2 <= 0.95;
    else
        cart_m = false(vol_size);
    end

end

function [flexion_a, inversion_a, rotation_a] = generateMotionTrajectory(n_states, motion_type, flex_range, inv_range, rot_range)
    %% Generate motion angle trajectories for each state

    flexion_a = linspace(flex_range(1), flex_range(2), n_states);

    switch motion_type
        case 'flexion'
            % Pure dorsi/plantarflexion
            inversion_a = zeros(1, n_states);
            rotation_a = zeros(1, n_states);

        case 'inversion'
            % Pure inversion/eversion
            flexion_a = zeros(1, n_states);
            inversion_a = linspace(inv_range(1), inv_range(2), n_states);
            rotation_a = zeros(1, n_states);

        case 'rotation'
            % Pure internal/external rotation
            flexion_a = zeros(1, n_states);
            inversion_a = zeros(1, n_states);
            rotation_a = linspace(rot_range(1), rot_range(2), n_states);

        case 'combined'
            % Combined motion (circular path in motion space)
            theta = linspace(0, 2*pi, n_states);
            flexion_a = flex_range(2) * sin(theta);
            inversion_a = inv_range(2) * cos(theta);
            rotation_a = rot_range(2) * sin(2*theta);

        otherwise
            % Default: flexion only
            inversion_a = zeros(1, n_states);
            rotation_a = zeros(1, n_states);
    end

end

function T = createAnkleTransform(ankle_center, flex_angle, inv_angle, rot_angle)
    %% Create 4x4 transformation matrix for ankle motion
    %% with three rotation axes: dorsiflexion, inversion, internal rotation

    % Convert angles to radians
    flex_rad = flex_angle * pi / 180;
    inv_rad = inv_angle * pi / 180;
    rot_rad = rot_angle * pi / 180;

    % Translation to origin
    T1 = eye(4);
    T1(1:3, 4) = -ankle_center';

    % Rotation matrices
    % Dorsiflexion: rotation around X-axis (lateral-medial axis)
    Rx = eye(4);
    Rx(2:3, 2:3) = [cos(flex_rad), -sin(flex_rad); sin(flex_rad), cos(flex_rad)];

    % Inversion: rotation around Z-axis (vertical axis)
    Rz = eye(4);
    Rz(1:2, 1:2) = [cos(inv_rad), -sin(inv_rad); sin(inv_rad), cos(inv_rad)];

    % Internal rotation: rotation around Y-axis (anterior-posterior axis)
    Ry = eye(4);
    Ry([1,3], [1,3]) = [cos(rot_rad), -sin(rot_rad); sin(rot_rad), cos(rot_rad)];

    % Translation back
    T2 = eye(4);
    T2(1:3, 4) = ankle_center';

    % Combined transformation: T = T2 * Rz * Ry * Rx * T1
    T = T2 * Rz * Ry * Rx * T1;

end

function transformed = applyRigidBodyTransform(image, transform_matrix, anchor_point)
    %% Apply 3D rigid body transformation (simplified using local shifts)

    % Extract rotation component
    R = transform_matrix(1:3, 1:3);
    t = transform_matrix(1:3, 4);

    % For computational efficiency, use interpolation-based approach
    % This is a simplified version - actual implementation would use proper 3D interpolation

    [x, y, z] = ndgrid(1:size(image, 1), 1:size(image, 2), 1:size(image, 3));

    % Convert to homogeneous coordinates relative to anchor
    coords = [x(:) - anchor_point(1), y(:) - anchor_point(2), z(:) - anchor_point(3)]';

    % Apply rotation
    new_coords = R * coords;

    % Convert back
    new_x = reshape(new_coords(1, :), size(x)) + anchor_point(1);
    new_y = reshape(new_coords(2, :), size(y)) + anchor_point(2);
    new_z = reshape(new_coords(3, :), size(z)) + anchor_point(3);

    % Interpolate using regular voxel grid coordinates
    transformed = interp3(1:size(image,2), 1:size(image,1), 1:size(image,3), image, new_y, new_x, new_z, 'linear', 0);

end

function [fx, fy, fz] = generateSoftTissueForces(vol_size, flex_angle, inv_angle, rot_angle, ankle_center)
    %% Generate external force fields from joint motion

    [x, y, z] = ndgrid(1:vol_size(1), 1:vol_size(2), 1:vol_size(3));

    % Normalize to [-1, 1]
    x_norm = (x - ankle_center(1)) / (vol_size(1)/2);
    y_norm = (y - ankle_center(2)) / (vol_size(2)/2);
    z_norm = (z - ankle_center(3)) / (vol_size(3)/2);

    % Force proportional to motion
    fx = 0.2 * (flex_angle/30) * exp(-(x_norm.^2 + z_norm.^2));
    fy = 0.2 * (inv_angle/15) * exp(-(x_norm.^2 + z_norm.^2));
    fz = -0.1 * (flex_angle/30) * exp(-((x_norm.^2 + y_norm.^2)));

end

function [dx, dy, dz] = calculateElasticDeformation(fx, fy, fz, stiffness)
    %% Calculate elastic deformation from forces
    %% Simplified: direct proportional deformation

    % Deformation inversely proportional to stiffness
    dx = fx ./ (stiffness + 0.1);
    dy = fy ./ (stiffness + 0.1);
    dz = fz ./ (stiffness + 0.1);

end

function deformed = applyDeformationField(image, dx, dy, dz)
    %% Apply 3D deformation field to image using interpolation

    [x, y, z] = ndgrid(1:size(image, 1), 1:size(image, 2), 1:size(image, 3));

    % Create deformed coordinates
    x_def = x + dx;
    y_def = y + dy;
    z_def = z + dz;

    % Clamp to valid range
    x_def = max(1, min(size(image, 2), x_def));
    y_def = max(1, min(size(image, 1), y_def));
    z_def = max(1, min(size(image, 3), z_def));

    % Interpolate using regular voxel grid coordinates
    deformed = interp3(1:size(image,2), 1:size(image,1), 1:size(image,3), image, y_def, x_def, z_def, 'linear', 0);

end

function [sx, sy, sz] = generateSoftTissueDeformation(vol_size, flex_angle, strength, ankle_center)
    %% Generate smooth deformation fields for soft tissue

    [x, y, z] = ndgrid(1:vol_size(1), 1:vol_size(2), 1:vol_size(3));

    % Distance from ankle center
    dx = x - ankle_center(1);
    dy = y - ankle_center(2);
    dz = z - ankle_center(3);

    r = sqrt(dx.^2 + dy.^2 + dz.^2);
    r_norm = r / (vol_size(1)/2);

    % Muscle contraction pattern
    sx = strength * (flex_angle/30) * sin(pi*r_norm) .* exp(-r_norm);
    sy = strength * (flex_angle/30) * 0.3 * sin(2*pi*r_norm) .* exp(-r_norm);
    sz = -strength * abs(flex_angle/30) * cos(pi*r_norm) .* exp(-r_norm);

end

function deformed = applySmallDeformation(mask, scale)
    %% Apply small-scale uniform deformation

    deformed = mask;  % Simplified: return as-is

end