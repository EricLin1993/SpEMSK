function [images_combined, images, recon_time, save_time, eff_fov_expansion, coil_profiles] = tf_nufft_radial_recon_lep(signal, scaled_trajectory, base_resolution, fov_mm, varargin)
%TF_NUFFT_RADIAL_RECON_LEP Reconstruct 3-D radial MRI data with the local NUFFT pipeline.
%
%
% Authors: Enping Lin (lep), and Fatih Calakli
% Email: enping.lin@childrens.harvard.edu
%
% Copyright (c) 2026 Enping Lin, Fatih Calakli, and Simon K. Warfield.
% All rights reserved.
%
% This code is provided for research and educational purposes only.
%========================================================================


[fov_expansion_factor, down_sampling_factor, subsets, low_memory, ...
  primal, adaptive_combine, coil_profiles, dcf, recon_dir, ...
  save_subsets, subset_name_tag, render_mode, ...
  save_coils, coil_name_tag, save_images_num_threads, ...
  imageOrigin_xyz, hostVoxDim_mm, input_compressed, ...
  principal_signal, convert_subsets_to_nii, ...
  output_image_type, output_image_precision] = process_options(varargin, ...
  'fov_expansion_factor', 1.1, 'down_sampling_factor', 1.0, 'subsets', [], 'low_memory', false, ...
  'primal', false, 'adaptive_combine', 0, 'coil_profiles', [], 'dcf', [], 'recon_dir', '', ...
  'save_subsets', false, 'subset_name_tag', 'subset', 'render_mode', false, ...
  'save_coils', false, 'coil_name_tag', 'coil', 'save_images_num_threads', 4, ...
  'imageOrigin_xyz', repelem(-0.5*(fov_mm-fov_mm/base_resolution), 3), 'hostVoxDim_mm', repelem(fov_mm/base_resolution, 3), ...
  'input_compressed', false, 'principal_signal', [], ...
  'convert_subsets_to_nii', false, 'output_image_type', [], 'output_image_precision', []);

assert(all(size(signal, 1:2)==size(scaled_trajectory, 1:2)), "signal and trajectory must have first two dimensions ==> nReadout x nSpokes");
if max(abs(scaled_trajectory), [], 'all') > 0.5
  scaled_trajectory = 0.5*scaled_trajectory./max(abs(scaled_trajectory(:)));
end

if isempty(subsets)
  subsets = {1:size(scaled_trajectory, 2)};
end

[subset_recon_dir, coil_recon_dir, render_mode] = prepare_output_dirs(recon_dir, save_subsets, save_coils, render_mode);
render_safe = save_subsets && render_mode;

final_image_size = repelem(ceil(base_resolution/down_sampling_factor), 3);
images_combined = allocate_combined_image(final_image_size, numel(subsets), render_safe, adaptive_combine, coil_profiles);

if adaptive_combine > 0 && low_memory
  normalizer = zeros(size(images_combined));
  kernel = ones(adaptive_combine, adaptive_combine, adaptive_combine);
  kernel = kernel/sum(kernel, 'all');
else
  normalizer = [];
  kernel = [];
end

subset_recon_time = zeros(1, numel(subsets));
subset_export_time = zeros(1, numel(subsets));
coil_export_time = zeros(1, size(signal, 3));

change_precision = @(x) x;
if strcmp(output_image_precision, 'single')
  change_precision = @(x) single(x);
end

change_type = @(x) x;
if strcmp(output_image_type, 'magnitude')
  change_type = @(x) abs(x);
end

images = [];
rs = 1;

for s = 1:numel(subsets)
  if ~render_safe
    rs = s;
  end

  subset_recon_timer = tic;
  fprintf('Truely Fast NUFFT reconstructing SUBSET %d/%d...\n', s, numel(subsets));

  subset_spokes = subsets{s};
  fprintf('Creating k-space interpolator...\n');
  PHI = build_sinc_interpolator(scaled_trajectory(:, subset_spokes, :), base_resolution, fov_mm, fov_expansion_factor, down_sampling_factor);

  image_size = repelem(round(size(PHI, 2)^(1/3)), 3);
  to_image = @(x) reshape(x, [image_size size(x, ndims(x))]);
  inverse_fft = @(x) n_dimensional_multi_channel_ifft(x, image_size);

  tic;
  if primal
    fprintf('Calculating voxelwise density compensation factor...\n');
    voxeldcf = calculate_voxelwise_dcf(PHI);
    iNUFFT = @(x) inverse_fft(to_image(voxeldcf.*(PHI'*x)));
  else
    if isempty(dcf)
      fprintf('Calculating pointwise density compensation factor...\n');
      pointdcf = calculate_pointwise_dcf(PHI);
    else
      fprintf('Using user provided pointwise density compensation factor...\n');
      pointdcf = reshape(dcf(:, subset_spokes), [], 1);
    end
    iNUFFT = @(x) inverse_fft(to_image(PHI'*(pointdcf.*x)));
  end
  toc;

  new_pixels = round((image_size-final_image_size)/2);
  selected_indices = (1+new_pixels(1)):(image_size(1)-new_pixels(1));
  phase_correction_image = reconstruct_phase_reference(iNUFFT, principal_signal, subset_spokes, selected_indices, input_compressed);

  tic;
  if low_memory
    [images_combined, low_memory_state] = reconstruct_low_memory(signal, subset_spokes, iNUFFT, selected_indices, rs, ...
      images_combined, adaptive_combine, phase_correction_image, kernel, normalizer, ...
      save_coils, s, numel(subsets), coil_recon_dir, coil_name_tag, change_precision, ...
      imageOrigin_xyz, down_sampling_factor*hostVoxDim_mm, coil_export_time);
    coil_export_time = low_memory_state.coil_export_time;
    normalizer = low_memory_state.normalizer;
    images = [];
  else
    [images, images_combined] = reconstruct_all_channels(signal, subset_spokes, iNUFFT, selected_indices, rs, ...
      images_combined, coil_profiles, coil_name_tag);
  end
  toc;

  subset_recon_time(s) = toc(subset_recon_timer);
  fprintf('Truely Fast NUFFT reconstructed SUBSET %d/%d in %.2f seconds!\n', s, numel(subsets), subset_recon_time(s));

  if render_safe
    batch_subset_export_timer = tic;
    fprintf('Writing %s image %d/%d...\n', subset_name_tag, s, numel(subsets));
    write_3D_image_to_disk(sprintf('%s%s%03d.nrrd', subset_recon_dir, subset_name_tag, s), change_precision(change_type(images_combined)), ...
      'imageOrigin_xyz', imageOrigin_xyz, 'hostVoxDim_mm', down_sampling_factor*hostVoxDim_mm);
    subset_export_time(s) = toc(batch_subset_export_timer);
    images_combined = zeros(size(images_combined), 'like', images_combined);
    if exist('normalizer', 'var')
      normalizer = zeros(size(normalizer));
    end
  end
end

recon_time = sum(subset_recon_time)-sum(coil_export_time);

if save_coils && ~low_memory
  batch_coil_export_timer = tic;
  write_images_to_disk(coil_recon_dir, change_precision(images), ...
    'imageOrigin_xyz', imageOrigin_xyz, ...
    'hostVoxDim_mm', down_sampling_factor*hostVoxDim_mm, ...
    'name_tag', coil_name_tag, 'num_threads', save_images_num_threads);
  coil_export_time = toc(batch_coil_export_timer);
end

if save_subsets && ~render_mode
  nifti_info = build_nifti_info(convert_subsets_to_nii, images_combined, down_sampling_factor*hostVoxDim_mm, imageOrigin_xyz, output_image_precision);
  batch_subset_export_timer = tic;
  write_images_to_disk(subset_recon_dir, change_precision(change_type(images_combined)), ...
    'imageOrigin_xyz', imageOrigin_xyz, ...
    'hostVoxDim_mm', down_sampling_factor*hostVoxDim_mm, ...
    'name_tag', subset_name_tag, ...
    'num_threads', save_images_num_threads, ...
    'nifti_info', nifti_info);
  subset_export_time = toc(batch_subset_export_timer);
end

save_time = sum(coil_export_time) + sum(subset_export_time);
eff_fov_expansion = image_size(1)/(image_size(1)-2*new_pixels(1));

end

function [subset_recon_dir, coil_recon_dir, render_mode] = prepare_output_dirs(recon_dir, save_subsets, save_coils, render_mode)
% Validate optional output folders and mirror the original directory layout.
subset_recon_dir = '';
coil_recon_dir = '';

if save_subsets
  if ~isfolder(recon_dir)
    error('%s is not accessible - provide proper path to save subset images!\n', recon_dir);
  end
  recon_dir = append_slash(recon_dir);
  subset_recon_dir = strcat(recon_dir, 'subsets/');
  make_directory(subset_recon_dir);
elseif render_mode
  warning('Turning off render_mode as save_subsets=false');
  render_mode = false;
end

if save_coils
  if ~isfolder(recon_dir)
    error('%s is not accessible - provide proper path to save coil images!\n', recon_dir);
  end
  recon_dir = append_slash(recon_dir);
  coil_recon_dir = strcat(recon_dir, 'coils/');
  make_directory(coil_recon_dir);
end
end

function path = append_slash(path)
if path(end) ~= '/'
  path(end+1) = '/';
end
end

function images_combined = allocate_combined_image(final_image_size, num_subsets, render_safe, adaptive_combine, coil_profiles)
% Match the original real/complex allocation rule for combined images.
if render_safe
  num_outputs = 1;
else
  num_outputs = num_subsets;
end

if adaptive_combine > 0 || ~isempty(coil_profiles)
  images_combined = complex(zeros([final_image_size num_outputs]));
else
  images_combined = zeros([final_image_size num_outputs]);
end
end

function phase_correction_image = reconstruct_phase_reference(iNUFFT, principal_signal, subset_spokes, selected_indices, input_compressed)
% Reconstruct the optional principal-signal phase reference.
if ~input_compressed && ~isempty(principal_signal)
  fprintf('Reconstructing principal signal\n');
  phase_correction_image = iNUFFT(vectorize_1d(principal_signal(:, subset_spokes)));
  phase_correction_image = phase_correction_image(selected_indices, selected_indices, selected_indices);
  phase_correction_image = exp(-1j*angle(phase_correction_image));
else
  phase_correction_image = [];
end
end

function [images_combined, state] = reconstruct_low_memory(signal, subset_spokes, iNUFFT, selected_indices, rs, ...
  images_combined, adaptive_combine, phase_correction_image, kernel, normalizer, ...
  save_coils, subset_index, num_subsets, coil_recon_dir, coil_name_tag, change_precision, ...
  imageOrigin_xyz, hostVoxDim_mm, coil_export_time)
% Reconstruct one channel at a time to reduce peak memory use.
fprintf('Memory-efficient mode! Reconstructing one channel at a time!\n');

for c = 1:size(signal, 3)
  fprintf('Reconstructing %s %d/%d...\n', coil_name_tag, c, size(signal, 3));
  image = iNUFFT(vectorize_1d(signal(:, subset_spokes, c)));
  image = image(selected_indices, selected_indices, selected_indices);

  if save_coils && subset_index == num_subsets
    coil_export_timer = tic;
    fprintf('Writing %s image %d/%d...\n', coil_name_tag, c, size(signal, 3));
    write_3D_image_to_disk(sprintf('%s%s%03d.nrrd', coil_recon_dir, coil_name_tag, c), change_precision(image), ...
      'imageOrigin_xyz', imageOrigin_xyz, 'hostVoxDim_mm', hostVoxDim_mm);
    coil_export_time(c) = toc(coil_export_timer);
  end

  if adaptive_combine > 0
    if c == 1 && isempty(phase_correction_image)
      phase_correction_image = exp(-1j*angle(image));
    end
    image_smoothed = imfilter(image.*phase_correction_image, kernel, 'replicate');
    images_combined(:, :, :, rs) = images_combined(:, :, :, rs) + conj(image_smoothed).*image;
    normalizer(:, :, :, rs) = normalizer(:, :, :, rs) + abs(image_smoothed).^2;
  else
    images_combined(:, :, :, rs) = images_combined(:, :, :, rs) + abs(image).^2;
  end
end

if adaptive_combine > 0
  images_combined(:, :, :, rs) = images_combined(:, :, :, rs)./sqrt(normalizer(:, :, :, rs));
else
  images_combined(:, :, :, rs) = sqrt(images_combined(:, :, :, rs));
end

state.normalizer = normalizer;
state.coil_export_time = coil_export_time;
end

function [images, images_combined] = reconstruct_all_channels(signal, subset_spokes, iNUFFT, selected_indices, rs, ...
  images_combined, coil_profiles, coil_name_tag)
% Reconstruct all channels together when memory is sufficient.
if size(signal, 3) > 1
  fprintf('Reconstructing %d %ss...\n', size(signal, 3), coil_name_tag);
  images = iNUFFT(vectorize_channels(signal(:, subset_spokes, :)));
  fprintf('Fixing FOV and creating a combined image...\n');
  images = images(selected_indices, selected_indices, selected_indices, :);
  images_combined(:, :, :, rs) = combine_channels(images, coil_profiles);
else
  fprintf('Reconstructing (one!) %s...\n', coil_name_tag);
  images = iNUFFT(vectorize_1d(signal(:, subset_spokes, :)));
  fprintf('Fixing FOV and creating a combined image...\n');
  images = images(selected_indices, selected_indices, selected_indices, :);
  images_combined(:, :, :, rs) = images;
end
end

function out = combine_channels(images, coil_profiles)
% Use supplied coil profiles when available; otherwise perform SOS combine.
if ~isempty(coil_profiles)
  out = sum(conj(coil_profiles).*images, ndims(images))./sqrt(sum(abs(coil_profiles).^2, ndims(images)));
else
  out = sqrt(sum(abs(images).^2, ndims(images)));
end
end

function x = vectorize_channels(x)
x = reshape(x, [], size(x, ndims(x)));
end

function x = vectorize_1d(x)
x = reshape(double(x), [], 1);
end

function PHI = build_sinc_interpolator(trajectory, base_resolution, fov, fov_expansion_factor, down_sampling_factor)
% Build the sinc interpolation matrix used by the radial inverse NUFFT.
if ~isa(trajectory, 'double')
  trajectory = double(trajectory);
end

dim = size(trajectory, ndims(trajectory));
fov_expansion_factor = max([1 base_resolution/fov fov_expansion_factor]);
new_pixels = ceil(0.5*(fov_expansion_factor-1)*base_resolution);
image_size = base_resolution + 2*new_pixels;
fov_expansion_factor = image_size/base_resolution;

disp(['base_resolution: ', num2str(base_resolution), ' fov_expansion_factor: ', num2str(fov_expansion_factor), ' down_sampling_factor: ', num2str(down_sampling_factor)]);

k_vec = reshape(trajectory, [], dim);
k_max = max(abs(k_vec), [], 'all');
assert(k_max <= 0.5, 'trajectory coordinates have values |k| > 0.5');
if k_max < 0.4
  warning('Did you properly scale the trajectory? k_max=%.2f', k_max);
end

original_num_samples = size(k_vec, 1);
sample_rows = (1:original_num_samples)';

if down_sampling_factor > 1
  keep_samples = ~any(abs(k_vec*down_sampling_factor) > 0.5, 2);
  k_vec = k_vec(keep_samples, :);
  sample_rows = sample_rows(keep_samples);
end

k_vec = (0.5+k_vec)*image_size;
kint_vec = floor(k_vec);
sub_to_lin_index = image_size.^(0:(dim-1))';
shifts = search_space(dim, 0.5);

num_samples = size(k_vec, 1);
num_shifts = size(shifts, 1);
non_uniform_indices = repmat(sample_rows, 1, num_shifts);
cartesian_indices = zeros(num_samples, num_shifts);
interpolation_weights = zeros(num_samples, num_shifts);

disp('generating sinc interpolation weights');
diff = kint_vec-k_vec;
for i = 1:num_shifts
  cartesian_indices(:, i) = 1 + mod(kint_vec+shifts(i, :), image_size)*sub_to_lin_index;
  interpolation_weights(:, i) = prod(sinc((diff+shifts(i, :))/fov_expansion_factor).*(abs((diff+shifts(i, :))/fov_expansion_factor) <= 1), 2);
end

logical_indices = abs(interpolation_weights) > eps('double');
disp(['dropping out weights that are below ' num2str(eps('double')) ' tolerance']);
disp(['creating a sparse matrix of size ', num2str(original_num_samples), 'x', num2str(image_size^dim), ' with ', num2str(nnz(logical_indices)), ' nonzeros.']);

PHI = sparse(non_uniform_indices(logical_indices), cartesian_indices(logical_indices), interpolation_weights(logical_indices), original_num_samples, image_size^dim);

if down_sampling_factor > 1
  keep_cartesian = downsampled_cartesian_mask(image_size, dim, down_sampling_factor);
  PHI = PHI(:, keep_cartesian);
end
end

function keep_cartesian = downsampled_cartesian_mask(image_size, dim, down_sampling_factor)
% Keep the center crop of the expanded Cartesian grid after downsampling.
image_center = floor(image_size/2)+1;
image_half_size = ceil(0.5*image_size/down_sampling_factor);
image_idx = [image_center-image_half_size image_center+image_half_size-1];

dummy_image = true(repelem(image_size, dim));
if dim == 2
  dummy_image(image_idx(1):image_idx(2), image_idx(1):image_idx(2)) = false;
else
  dummy_image(image_idx(1):image_idx(2), image_idx(1):image_idx(2), image_idx(1):image_idx(2)) = false;
end
keep_cartesian = find(~dummy_image(:));
end

function pointdcf = calculate_pointwise_dcf(interpolator, Gram)
if nargin < 2
  Gram = interpolator'*interpolator;
end

numer = full(sum(interpolator.*conj(interpolator), 2));
denom = full(sum((interpolator*Gram).*conj(interpolator), 2));
pointdcf = real(numer./denom);
pointdcf(isnan(pointdcf)) = 0;
pointdcf(isinf(pointdcf)) = 0;
end

function voxeldcf = calculate_voxelwise_dcf(interpolator, Gram, lambda)
if nargin < 3
  lambda = 0;
end
if nargin < 2
  Gram = interpolator'*interpolator;
end

numer = full(diag(Gram));
denom = full(sum(Gram.*Gram, 1))';
if lambda > 0
  numer = numer + lambda;
  denom = denom + 2*lambda*numer + lambda^2;
end

voxeldcf = numer./denom;
voxeldcf(isnan(voxeldcf)) = 0;
voxeldcf(isinf(voxeldcf)) = 0;
end

function out = n_dimensional_multi_channel_ifft(in, image_size, orthogonal)
if nargin < 3
  orthogonal = true;
end

original_size = size(in);
num_channel = prod(original_size)/prod(image_size);
assert((num_channel >= 1) && (num_channel == floor(num_channel)), 'input image size is possibly wrong');

out = reshape(in, [image_size num_channel]);
if orthogonal
  for dim = 1:numel(image_size)
    out = fftshift(ifft(ifftshift(out, dim), [], dim), dim)*sqrt(image_size(dim));
  end
else
  for dim = 1:numel(image_size)
    out = fftshift(ifft(ifftshift(out, dim), [], dim), dim);
  end
end
out = reshape(out, original_size);
end

function shifts = search_space(dim, radius)
if nargin < 2
  radius = 0.5;
end
if nargin < 1
  dim = 2;
end

shifts = dec2base(0:4^dim-1, 4)-'0'-1;
bits = dec2bin(0:2^dim-1)-'0';
min_dist = inf(size(shifts, 1), 1);
for is = 1:size(shifts, 1)
  for ib = 1:size(bits, 1)
    min_dist(is) = min(min_dist(is), norm(shifts(is, :)-bits(ib, :)));
  end
end
shifts = shifts(min_dist <= radius, :);
end

function nifti_info = build_nifti_info(convert_subsets_to_nii, images_combined, pixel_dimensions, imageOrigin_xyz, output_image_precision)
% Create the same minimal NIfTI metadata block used by the original code.
nifti_info = [];
if ~convert_subsets_to_nii
  return;
end

nifti_info.Version = 'NIfTI1';
nifti_info.Description = 'Created in MATLAB R2022b';
nifti_info.ImageSize = size(images_combined, 1:3);
nifti_info.PixelDimensions = pixel_dimensions;
nifti_info.Datatype = 'double';
nifti_info.BitsPerPixel = 64;
if strcmp(output_image_precision, 'single')
  nifti_info.Datatype = 'single';
  nifti_info.BitsPerPixel = 32;
end
nifti_info.SpaceUnits = 'Millimeter';
nifti_info.Qfactor = 1;
nifti_info.TimeUnits = 'None';
nifti_info.SliceCode = 'Unknown';
nifti_info.AdditiveOffset = 0;
nifti_info.MultiplicativeScaling = 1;
nifti_info.TimeOffset = 0;
nifti_info.FrequencyDimension = 0;
nifti_info.PhaseDimension = 0;
nifti_info.SpatialDimension = 0;
nifti_info.DisplayIntensityRange = [0 0];
nifti_info.TransformName = 'Qform';
nifti_info.Transform.T = [diag(diag([-1 -1 1])*nifti_info.PixelDimensions') zeros(3, 1); imageOrigin_xyz*diag([-1 -1 1]) 1];
nifti_info.Transform.Dimensionality = 3;
end

function [varargout] = process_options(args, varargin)
% Lightweight name-value parser compatible with the original helper.
n = length(varargin);
if mod(n, 2)
  error('Each option must be a string/value pair.');
end

if nargout < (n/2)
  error('Insufficient number of output arguments given');
elseif nargout == (n/2)
  warn = 1;
  nout = n/2;
else
  warn = 0;
  nout = n/2 + 1;
end

varargout = cell(1, nout);
for i = 2:2:n
  varargout{i/2} = varargin{i};
end

nunused = 0;
unused = {};
for i = 1:2:length(args)
  found = 0;
  for j = 1:2:n
    if strcmpi(args{i}, varargin{j})
      varargout{(j+1)/2} = args{i+1};
      found = 1;
      break;
    end
  end
  if ~found
    if warn
      warning('Option ''%s'' not used.', args{i});
    else
      nunused = nunused + 1;
      unused{2*nunused-1} = args{i};
      unused{2*nunused} = args{i+1};
    end
  end
end

if ~warn
  if nunused
    varargout{nout} = unused;
  else
    varargout{nout} = cell(0);
  end
end
end

function make_directory(dirname, verbose)
if nargin < 2
  verbose = true;
end

if ~isfolder(dirname)
  if verbose
    fprintf('Creating directory %s\n', dirname);
  end
  mkdir(dirname);
elseif verbose
  fprintf('%s already exists\n', dirname);
end
end

function write_images_to_disk(path, images, varargin)
[imageOrigin_xyz, hostVoxDim_mm, name_tag, num_threads, nifti_info] = process_options(varargin, ...
  'imageOrigin_xyz', -0.5*size(images, 1:3), ...
  'hostVoxDim_mm', ones(1, 3), ...
  'name_tag', 'coil', ...
  'num_threads', 4, ...
  'nifti_info', []);

if ~isfolder(path)
  error('%s is not accessible - check the path!\n', path);
end
path = append_slash(path);

num_images = size(images, 4);
fprintf('Writing %d %s images to disk...\n', num_images, name_tag);
if num_threads > 1
  parpoolobj = gcp('nocreate');
  if isempty(parpoolobj)
    parpool('Threads', num_threads);
  end
  parfor i = 1:num_images
    fprintf('Writing %s image %d/%d...\n', name_tag, i, num_images);
    fname = sprintf('%s%s%03d.nrrd', path, name_tag, i);
    write_3D_image_to_disk(fname, images(:, :, :, i), 'hostVoxDim_mm', hostVoxDim_mm, 'imageOrigin_xyz', imageOrigin_xyz, 'nifti_info', nifti_info);
  end
  delete(gcp('nocreate'));
else
  for i = 1:num_images
    fprintf('Writing %s image %d/%d...\n', name_tag, i, num_images);
    fname = sprintf('%s%s%03d.nrrd', path, name_tag, i);
    write_3D_image_to_disk(fname, images(:, :, :, i), 'hostVoxDim_mm', hostVoxDim_mm, 'imageOrigin_xyz', imageOrigin_xyz, 'nifti_info', nifti_info);
  end
end
end

function ok = write_3D_image_to_disk(path, image, varargin)
assert(ndims(image) == 3, 'This function is designed to write a 3D image to disk (complex or real)!');
[imageOrigin_xyz, hostVoxDim_mm, nifti_info] = process_options(varargin, ...
  'imageOrigin_xyz', -0.5*(size(image)-1), ...
  'hostVoxDim_mm', ones(1, 3), ...
  'nifti_info', []);

if isreal(image)
  if ~isempty(nifti_info)
    niftiwrite(permute(image, [2 1 3]), strrep(path, '.nrrd', '.nii'), nifti_info);
  end
  ok = nrrdWriter(path, image, hostVoxDim_mm, imageOrigin_xyz, 'raw');
else
  if ~isempty(nifti_info)
    niftiwrite(permute(abs(image), [2 1 3]), strrep(path, '.nrrd', '.nii'), nifti_info);
  end
  ok = nrrdWriterComplex3d(path, image, hostVoxDim_mm, imageOrigin_xyz, 'raw');
end
end

function ok = nrrdWriter(filename, matrix, pixelspacing, origin, encoding)
[pathf, fname, ext] = fileparts(filename);
format = ext(2:end);

dims = size(matrix);
nd = length(dims);
if isequal(nd, 3) && isequal(dims(1), 2)
  matrix = permute(matrix, [1 3 2]);
elseif isequal(nd, 3) && dims(1) > 2
  matrix = permute(matrix, [2 1 3]);
else
  matrix = permute(matrix, [1 3 2 4]);
end

dims = size(matrix);
nd = length(dims);
encoding = lower(encoding);
format = lower(format);
assert(isequal(encoding, 'ascii') || isequal(encoding, 'raw') || isequal(encoding, 'gzip'), 'Unsupported encoding');
assert(isequal(format, 'nhdr') || isequal(format, 'nrrd'), 'Unexpected format');

fid = fopen(filename, 'wb');
fprintf(fid, 'NRRD0004\n');
outtype = nrrd_datatype(class(matrix));
fprintf(fid, ['type: ', outtype, '\n']);
fprintf(fid, ['dimension: ', num2str(nd), '\n']);

if isequal(nd, 2)
  fprintf(fid, 'space: left-posterior\n');
elseif isequal(nd, 3) && isequal(dims(1), 2)
  fprintf(fid, 'space: none-left-posterior\n');
else
  fprintf(fid, 'space: left-posterior-superior\n');
end

fprintf(fid, ['sizes: ', num2str(dims), '\n']);
write_nrrd_geometry(fid, nd, dims, pixelspacing, false);
fprintf(fid, ['encoding: ', encoding, '\n']);
write_nrrd_endian(fid);
write_nrrd_origin(fid, nd, dims, origin, false);
fid = switch_to_nhdr_payload_if_needed(fid, format, pathf, fname, encoding);
ok = write_nrrd_data(fid, matrix, outtype, encoding);
fclose(fid);
end

function ok = nrrdWriterComplex3d(filename, matrix, pixelspacing, origin, encoding)
matrix = cat(4, real(matrix), imag(matrix));
matrix = permute(matrix, [4 1 2 3]);

[pathf, fname, ext] = fileparts(filename);
format = ext(2:end);

dims = size(matrix);
nd = length(dims);
if isequal(nd, 3) && isequal(dims(1), 2)
  matrix = permute(matrix, [1 3 2]);
elseif isequal(nd, 3) && dims(1) > 2
  matrix = permute(matrix, [2 1 3]);
else
  matrix = permute(matrix, [1 3 2 4]);
end

dims = size(matrix);
nd = length(dims);
if isequal(nd, 3) && isequal(dims(1), 2)
  dims = [dims, 1];
  nd = nd + 1;
end

encoding = lower(encoding);
format = lower(format);
assert(isequal(encoding, 'ascii') || isequal(encoding, 'raw') || isequal(encoding, 'gzip'), 'Unsupported encoding');
assert(isequal(format, 'nhdr') || isequal(format, 'nrrd'), 'Unexpected format');

fid = fopen(filename, 'wb');
fprintf(fid, 'NRRD0004\n');
outtype = nrrd_datatype(class(matrix));
fprintf(fid, ['type: ', outtype, '\n']);
fprintf(fid, ['dimension: ', num2str(nd), '\n']);
fprintf(fid, 'space: left-posterior-superior\n');
fprintf(fid, ['sizes: ', num2str(dims), '\n']);
write_nrrd_geometry(fid, nd, dims, pixelspacing, true);
fprintf(fid, ['encoding: ', encoding, '\n']);
write_nrrd_endian(fid);
fprintf(fid, ['space origin: (', num2str(origin(1)), ',', num2str(origin(2)), ',', num2str(origin(3)), ')\n']);
fid = switch_to_nhdr_payload_if_needed(fid, format, pathf, fname, encoding);
ok = write_nrrd_data(fid, matrix, outtype, encoding);
fclose(fid);
end

function datatype = nrrd_datatype(metaType)
switch metaType
  case {'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'double'}
    datatype = metaType;
  case 'single'
    datatype = 'float';
  otherwise
    assert(false, 'Unknown datatype');
end
end

function write_nrrd_geometry(fid, nd, dims, pixelspacing, force_3d_vector)
if isequal(nd, 2)
  fprintf(fid, ['space directions: (', num2str(pixelspacing(1)), ',0) (0,', num2str(pixelspacing(2)), ')\n']);
  fprintf(fid, 'kinds: domain domain\n');
elseif isequal(nd, 3) && isequal(dims(1), 2) && ~force_3d_vector
  fprintf(fid, ['space directions: none (', num2str(pixelspacing(1)), ',0) (0,', num2str(pixelspacing(2)), ')\n']);
  fprintf(fid, 'kinds: vector domain domain\n');
elseif isequal(nd, 3) && dims(1) > 2
  fprintf(fid, ['space directions: (', num2str(pixelspacing(1)), ',0,0) (0,', num2str(pixelspacing(2)), ',0) (0,0,', num2str(pixelspacing(3)), ')\n']);
  fprintf(fid, 'kinds: domain domain domain\n');
else
  fprintf(fid, ['space directions: none (', num2str(pixelspacing(1)), ',0,0) (0,', num2str(pixelspacing(2)), ',0) (0,0,', num2str(pixelspacing(3)), ')\n']);
  fprintf(fid, 'kinds: vector domain domain domain\n');
end
end

function write_nrrd_endian(fid)
[~, ~, endian] = computer();
if isequal(endian, 'B')
  fprintf(fid, 'endian: big\n');
else
  fprintf(fid, 'endian: little\n');
end
end

function write_nrrd_origin(fid, nd, dims, origin, force_3d_vector)
if isequal(nd, 2) || (isequal(nd, 3) && isequal(dims(1), 2) && ~force_3d_vector)
  fprintf(fid, ['space origin: (', num2str(origin(1)), ',', num2str(origin(2)), ')\n']);
else
  fprintf(fid, ['space origin: (', num2str(origin(1)), ',', num2str(origin(2)), ',', num2str(origin(3)), ')\n']);
end
end

function fid = switch_to_nhdr_payload_if_needed(fid, format, pathf, fname, encoding)
if isequal(format, 'nhdr')
  fprintf(fid, ['data file: ', [fname, '.', encoding], '\n']);
  fclose(fid);
  if isequal(length(pathf), 0)
    fid = fopen([fname, '.', encoding], 'wb');
  else
    fid = fopen([pathf, filesep, fname, '.', encoding], 'wb');
  end
else
  fprintf(fid, '\n');
end
end

function ok = write_nrrd_data(fidIn, matrix, datatype, encoding)
switch encoding
  case 'raw'
    ok = fwrite(fidIn, matrix(:), datatype);
  case 'gzip'
    tmpBase = tempname(pwd);
    tmpFile = [tmpBase '.gz'];
    fidTmpRaw = fopen(tmpBase, 'wb');
    assert(fidTmpRaw > 3, 'Could not open temporary file for GZIP compression');
    fwrite(fidTmpRaw, matrix(:), datatype);
    fclose(fidTmpRaw);
    gzip(tmpBase);
    fidTmpRaw = fopen(tmpFile, 'rb');
    cleaner = onCleanup(@() fclose(fidTmpRaw));
    tmp = fread(fidTmpRaw, inf, [datatype '=>' datatype]);
    ok = fwrite(fidIn, tmp, datatype);
    delete(tmpBase);
    delete(tmpFile);
  case 'ascii'
    ok = fprintf(fidIn, '%u ', matrix(:));
  otherwise
    assert(false, 'Unsupported encoding');
end
end
