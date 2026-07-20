function [images_combined, images, recon_time, save_time, eff_fov_expansion, coil_profiles] = nufft_Open(signal, scaled_trajectory, base_resolution, fov_mm, varargin)
%NUFFT_OPEN Self-contained open 3-D radial gridding NUFFT reconstruction.
%
% Drop-in replacement for tf_nufft_radial_recon_lep: same inputs, same
% name-value options, same six outputs, so an existing call is switched over by
% changing ONLY the function name:
%
%   img = tf_nufft_radial_recon_lep(kSpace, traj, baseResolution, fovx_mm, ...);
%   img = nufft_Open(               kSpace, traj, baseResolution, fovx_mm, ...);
%
% ------------------------------------------------------------------------
% PROVENANCE / LICENSING NOTE
% ------------------------------------------------------------------------
% Every algorithm in this file is a re-implementation of a PUBLISHED, publicly
% available method (see the reference list below).  NO in-house or proprietary
% code from our group is used or reproduced here.  In particular the density
% compensation is re-derived from the published least-squares / point-spread-
% function formulation of gridding (Rosenfeld 1998; Sedarat & Nishimura 2000;
% Zheng & Kaveh 2005; Dwork et al. 2021), not from any internal implementation.
% This file is therefore safe to release publicly.
%
% ------------------------------------------------------------------------
% METHOD
% ------------------------------------------------------------------------
% Classical Kaiser-Bessel gridding NUFFT [1,2]:
%   1. density compensation of the non-Cartesian samples (see below),
%   2. sparse convolution ("gridding") onto an oversampled Cartesian grid,
%   3. inverse FFT,
%   4. de-apodization by the analytic Fourier transform of the kernel,
%   5. cropping of the FOV-expanded grid back to the requested matrix size.
% The interpolation operator is assembled ONCE per subset as a sparse matrix
% PHI, so gridding all coils is a single sparse matrix product (fast).
%
% DENSITY COMPENSATION ('dcf_method'):
%   'lsq'    (default) One-shot LEAST-SQUARES / PSF-matched weights [3,4,5,6]:
%            PRIOR ART: this is the point-spread-function-modelling gridding DCF
%            of Samsonov, Kholmovski & Johnson (ISMRM 2003, p.477) [5], resting
%            on the least-squares formulation of gridding by Rosenfeld (1998)
%            [3] and Sedarat & Nishimura (2000) [4].  It is NOT a new method.
%            with A = PHI*PHI' (the sample-to-sample point-spread operator),
%                w_m = argmin_alpha || alpha*A(:,m) - e_m ||^2
%                    = A_mm / (A^2)_mm
%                    = diag(PHI*PHI') ./ diag(PHI*(PHI'*PHI)*PHI')
%            i.e. each sample is scaled so that its point spread function best
%            approximates a delta in the least-squares sense.  Closed form, no
%            iterations - this is why it is fast.
%   'pipe'   Iterative Pipe & Menon fixed point w <- w./(A*w) [7].
%   'radial' Analytic 3-D radial shell weights (see dcf_radial_shell).
%
% ------------------------------------------------------------------------
% REFERENCES
% ------------------------------------------------------------------------
% [1] J. I. Jackson, C. H. Meyer, D. G. Nishimura, A. Macovski, "Selection of a
%     convolution function for Fourier inversion using gridding," IEEE Trans.
%     Med. Imaging, 10(3):473-478, 1991.
% [2] P. J. Beatty, D. G. Nishimura, J. M. Pauly, "Rapid gridding reconstruction
%     with a minimal oversampling ratio," IEEE Trans. Med. Imaging,
%     24(6):799-808, 2005.
% [3] D. Rosenfeld, "An optimal and efficient new gridding algorithm using
%     singular value decomposition," Magn. Reson. Med., 40(1):14-23, 1998.
%     doi:10.1002/mrm.1910400103
% [4] H. Sedarat, D. G. Nishimura, "On the optimality of the gridding
%     reconstruction algorithm," IEEE Trans. Med. Imaging, 19(4):306-317, 2000.
%     doi:10.1109/42.848182            <-- formulates gridding as LEAST SQUARES
%                                          and derives the optimal weights from
%                                          the interpolation matrix; this is the
%                                          theoretical basis of the 'lsq' DCF.
%     Section III "Optimal Density Compensation" (p.308) derives the CLOSED-FORM
%     optimal DCF, Eq. (8):  d_ii = [T(T'T)^(-1)T']_ii / [T T']_ii, and the
%     Summary states "a closed form solution for the optimal DCF's is obtained".
%     See dcf_least_squares below for how the weights used here relate to it.
% [5] A. A. Samsonov, E. G. Kholmovski, C. R. Johnson, "Determination of the
%     sampling density compensation function using a point spread function
%     modeling approach and gridding approximation," Proc. ISMRM 11th Annual
%     Meeting, Toronto, 2003, p. 477.   <-- prior art for exactly this PSF-based
%                                          gridding DCF (cited as [14] by [6]).
% [6] N. Dwork, D. O'Connor, E. M. I. Johnson, C. A. Baron, J. W. Gordon,
%     J. M. Pauly, P. E. Z. Larson, "Least squares optimal density compensation
%     for the gridding non-uniform discrete Fourier transform,"
%     arXiv:2106.06660, 2021.  (See also ISMRM 2017, abstract 4009.)
% [7] J. G. Pipe, P. Menon, "Sampling density compensation in MRI: rationale and
%     an iterative numerical solution," Magn. Reson. Med., 41(1):179-186, 1999.
%     doi:10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V
% [8] K. O. Johnson, J. G. Pipe, "Convolution kernel design and efficient
%     algorithm for sampling density correction," Magn. Reson. Med.,
%     61(2):439-447, 2009.  doi:10.1002/mrm.21840
% [9] N. R. Zwart, K. O. Johnson, J. G. Pipe, "Efficient sample density
%     estimation by combining gridding and an optimized kernel," Magn. Reson.
%     Med., 67(3):701-710, 2012.
% [10] Y. Zheng, M. Kaveh, "Point spread function optimization for MRI
%     reconstruction," IEEE ICASSP, 2005.
%
% ------------------------------------------------------------------------
% INPUTS (identical to tf_nufft_radial_recon_lep)
%   signal            [nReadout x nSpokes x nCoils] complex k-space
%   scaled_trajectory [nReadout x nSpokes x 3]      k-space coords, |k| <= 0.5
%   base_resolution   scalar, reconstructed matrix size
%   fov_mm            scalar, field of view in mm
%
% NAME-VALUE OPTIONS (same names/defaults as the in-house version)
%   'fov_expansion_factor'  (1.1)   grid expansion / gridding oversampling
%   'down_sampling_factor'  (1.0)   reconstruct at reduced resolution
%   'subsets'               ({all}) cell array of spoke index vectors
%   'low_memory'            (false) do not keep the individual coil images
%   'adaptive_combine'      (0)     >0 = adaptive (Walsh-like) coil combine
%   'coil_profiles'         ([])    [Nx Ny Nz nCoils] sensitivity maps
%   'dcf'                   ([])    user-supplied density compensation
%   'principal_signal'      ([])    phase reference signal
%   'input_compressed'      (false)
%   The remaining options are accepted for API compatibility.
%
% ADDITIONAL (open-implementation specific) OPTIONS
%   'dcf_method'        ('lsq')  'lsq' | 'pipe' | 'radial'
%   'dcf_iters'         (10)     iterations when dcf_method = 'pipe'
%   'kernel'            ('sinc') 'sinc' | 'kb'
%                                'sinc': truncated sinc scaled by the FOV
%                                        expansion; its transform is ~rect over
%                                        the cropped FOV so NO de-apodization is
%                                        needed (classical band-limited gridding)
%                                'kb'  : Kaiser-Bessel [1,2] + analytic
%                                        de-apodization
%   'kernel_width'      (2)      stencil width per dimension (2 -> 2x2x2 = 8
%                                neighbours per sample, as in classical gridding)
%   'dcf_kernel_width'  ([])     kernel width used to build the LSQ DCF.
%                                [] = same as 'kernel_width'.  IMPORTANT: the
%                                least-squares DCF is derived for a GIVEN
%                                interpolation operator, so the DCF and the
%                                reconstruction should use the SAME kernel;
%                                a mismatch makes the weights sub-optimal.
%                                Note the LSQ Gram matrix grows as (2W-1)^3.
%
% OUTPUTS (identical to tf_nufft_radial_recon_lep)
%   images_combined   [Nx Ny Nz nSubsets] coil-combined image(s)
%   images            [Nx Ny Nz nCoils]   individual coil images (last subset)
%   recon_time        total reconstruction time in seconds
%   save_time         0 (see note)
%   eff_fov_expansion effective FOV expansion actually used
%   coil_profiles     the coil profiles that were used (pass-through)
%


% Author: Enping Lin Ph.D.,   enping.lin@chidlrens.harvard.edu
% 
%========================================================================

[fov_expansion_factor, down_sampling_factor, subsets, low_memory, ...
  primal, adaptive_combine, coil_profiles, dcf, recon_dir, ...
  save_subsets, subset_name_tag, render_mode, ...
  save_coils, coil_name_tag, save_images_num_threads, ...
  imageOrigin_xyz, hostVoxDim_mm, input_compressed, ...
  principal_signal, convert_subsets_to_nii, ...
  output_image_type, output_image_precision, ...
  dcf_method, dcf_iters, kernel_width, dcf_kernel_width, kernel_type] = process_options(varargin, ...
  'fov_expansion_factor', 1.1, 'down_sampling_factor', 1.0, 'subsets', [], 'low_memory', false, ...
  'primal', false, 'adaptive_combine', 0, 'coil_profiles', [], 'dcf', [], 'recon_dir', '', ...
  'save_subsets', false, 'subset_name_tag', 'subset', 'render_mode', false, ...
  'save_coils', false, 'coil_name_tag', 'coil', 'save_images_num_threads', 4, ...
  'imageOrigin_xyz', repelem(-0.5*(fov_mm-fov_mm/base_resolution), 3), 'hostVoxDim_mm', repelem(fov_mm/base_resolution, 3), ...
  'input_compressed', false, 'principal_signal', [], ...
  'convert_subsets_to_nii', false, 'output_image_type', [], 'output_image_precision', [], ...
  'dcf_method', 'lsq', 'dcf_iters', 10, 'kernel_width', 2, 'dcf_kernel_width', [], ...
  'kernel', 'sinc');

% The least-squares DCF is derived FOR A GIVEN interpolation operator, so by
% default the DCF is built with exactly the same kernel as the reconstruction
% (dcf_kernel_width = [] -> kernel_width).  Using a different kernel for the
% two makes the weights sub-optimal.
if isempty(dcf_kernel_width)
  dcf_kernel_width = kernel_width;
end

assert(all(size(signal, 1:2) == size(scaled_trajectory, 1:2)), ...
  'signal and trajectory must have first two dimensions ==> nReadout x nSpokes');

if max(abs(scaled_trajectory), [], 'all') > 0.5
  scaled_trajectory = 0.5*scaled_trajectory./max(abs(scaled_trajectory(:)));
end

if isempty(subsets)
  subsets = {1:size(scaled_trajectory, 2)};
end

if save_subsets || save_coils
  warning(['nufft_Open: ''save_subsets''/''save_coils'' are not supported in the ' ...
           'self-contained open implementation. Images are still returned in memory.']);
end

% ---- grid geometry: expand, reconstruct, then crop (as in the original) ----
fov_expansion_factor = max([1 base_resolution/fov_mm fov_expansion_factor]);
new_pixels_expand = ceil(0.5*(fov_expansion_factor-1)*base_resolution);
expanded_size = base_resolution + 2*new_pixels_expand;

if down_sampling_factor > 1
  grid_size = 2*ceil(0.5*expanded_size/down_sampling_factor);
else
  grid_size = expanded_size;
end
final_image_size = repelem(ceil(base_resolution/down_sampling_factor), 3);

fprintf('base_resolution: %d  fov_expansion_factor: %g  down_sampling_factor: %g\n', ...
  base_resolution, expanded_size/base_resolution, down_sampling_factor);

new_pixels = round((grid_size-final_image_size)/2);
selected_indices = (1+new_pixels(1)):(grid_size-new_pixels(1));

% ---- Kaiser-Bessel kernel parameters (Beatty et al. 2005 [2]) ----
W = kernel_width;
osf = grid_size/numel(selected_indices);
beta = kb_beta(W, osf);
fov_expansion_used = expanded_size/base_resolution;   % sinc kernel scale

if strcmpi(kernel_type, 'sinc')
  % The Fourier transform of sinc(u/F) is a rect of width 1/F, i.e. flat over
  % the cropped FOV -> no de-apodization needed (classical band-limited
  % interpolation gridding, and the same reason the in-house code omits it).
  apod3 = ones(final_image_size);
else
  apod1 = kb_deapodization(grid_size, W, beta);
  apod1 = apod1(selected_indices);
  apod3 = reshape(apod1, [], 1, 1).*reshape(apod1, 1, [], 1).*reshape(apod1, 1, 1, []);
  apod_floor = 1e-6*max(abs(apod3(:)));
  apod3(abs(apod3) < apod_floor) = apod_floor;
end

images_combined = allocate_combined_image(final_image_size, numel(subsets), adaptive_combine, coil_profiles);
subset_recon_time = zeros(1, numel(subsets));
images = [];

for s = 1:numel(subsets)
  subset_recon_timer = tic;
  fprintf('Open gridding NUFFT reconstructing SUBSET %d/%d...\n', s, numel(subsets));

  subset_spokes = subsets{s};
  traj_subset = double(scaled_trajectory(:, subset_spokes, :));
  k_vec = reshape(traj_subset, [], size(traj_subset, ndims(traj_subset)));

  if down_sampling_factor > 1
    keep_samples = ~any(abs(k_vec*down_sampling_factor) > 0.5, 2);
  else
    keep_samples = true(size(k_vec, 1), 1);
  end
  k_keep = k_vec(keep_samples, :);
  gk = (0.5 + (expanded_size/grid_size)*k_keep)*grid_size;   % grid units [0,G)

  % ---- sparse interpolation operator, built ONCE for this subset ----
  fprintf('Building sparse interpolator (%d samples, %d^3 grid, KB W=%d, beta=%.2f)...\n', ...
    size(gk, 1), grid_size, W, beta);
  tic;
  PHI = build_sparse_interpolator(gk, grid_size, W, beta, kernel_type, fov_expansion_used);
  toc;

  % ---- density compensation ----
  tic;
  if ~isempty(dcf)
    fprintf('Using user provided density compensation factor...\n');
    w = reshape(double(dcf(:, subset_spokes)), [], 1);
    w = w(keep_samples);
  else
    switch lower(dcf_method)
      case 'lsq'
        fprintf('Calculating one-shot least-squares (PSF-matched) DCF [3,4,5,6]...\n');
        if dcf_kernel_width == W
          PHI_dcf = PHI;
        else
          beta_dcf = kb_beta(dcf_kernel_width, osf);
          PHI_dcf = build_sparse_interpolator(gk, grid_size, dcf_kernel_width, beta_dcf, kernel_type, fov_expansion_used);
        end
        w = dcf_least_squares(PHI_dcf);
        clear PHI_dcf
      case 'pipe'
        fprintf('Calculating Pipe & Menon iterative DCF [7]...\n');
        w = dcf_pipe_menon(PHI, dcf_iters);
      otherwise
        fprintf('Calculating analytic 3-D radial shell DCF...\n');
        w_full = dcf_radial_shell(traj_subset);
        w = w_full(keep_samples);
    end
  end
  w = w/mean(w(w > 0));
  toc;

  % ---- optional phase reference from the principal signal ----
  phase_correction_image = [];
  if ~input_compressed && ~isempty(principal_signal)
    fprintf('Reconstructing principal signal\n');
    ps = reshape(double(principal_signal(:, subset_spokes)), [], 1);
    ps = ps(keep_samples);
    ref = ifft_crop_deapod(reshape(PHI'*(w.*ps), [grid_size grid_size grid_size]), ...
      selected_indices, apod3);
    phase_correction_image = exp(-1j*angle(ref));
  end

  % ---- gridding: ONE sparse product for all coils ----
  tic;
  num_coils = size(signal, 3);
  x = reshape(double(signal(:, subset_spokes, :)), [], num_coils);
  x = x(keep_samples, :);
  fprintf('Gridding %d %s(s) with a single sparse product...\n', num_coils, coil_name_tag);
  Gflat = PHI'*(w.*x);                       % [G^3 x nCoils]
  clear x PHI

  if ~low_memory
    images = complex(zeros([final_image_size num_coils]));
  end
  acc_sos = zeros(final_image_size);
  acc_num = complex(zeros(final_image_size));
  acc_den = zeros(final_image_size);
  if adaptive_combine > 0
    kernel = ones(adaptive_combine, adaptive_combine, adaptive_combine);
    kernel = kernel/sum(kernel(:));
  end

  for c = 1:num_coils
    img = ifft_crop_deapod(reshape(Gflat(:, c), [grid_size grid_size grid_size]), ...
      selected_indices, apod3);
    if ~low_memory
      images(:, :, :, c) = img;
    end
    if ~isempty(coil_profiles)
      acc_num = acc_num + conj(coil_profiles(:, :, :, c)).*img;
      acc_den = acc_den + abs(coil_profiles(:, :, :, c)).^2;
    elseif adaptive_combine > 0
      if c == 1 && isempty(phase_correction_image)
        phase_correction_image = exp(-1j*angle(img));
      end
      img_smoothed = box_filter3(img.*phase_correction_image, kernel);
      acc_num = acc_num + conj(img_smoothed).*img;
      acc_den = acc_den + abs(img_smoothed).^2;
    else
      acc_sos = acc_sos + abs(img).^2;
    end
  end
  clear Gflat

  if ~isempty(coil_profiles) || adaptive_combine > 0
    images_combined(:, :, :, s) = acc_num./sqrt(max(acc_den, eps));
  elseif num_coils > 1
    images_combined(:, :, :, s) = sqrt(acc_sos);
  else
    images_combined(:, :, :, s) = images(:, :, :, 1);
  end
  toc;

  subset_recon_time(s) = toc(subset_recon_timer);
  fprintf('Open gridding NUFFT reconstructed SUBSET %d/%d in %.2f seconds!\n', ...
    s, numel(subsets), subset_recon_time(s));
end

recon_time = sum(subset_recon_time);
save_time = 0;
eff_fov_expansion = grid_size/numel(selected_indices);

if strcmp(output_image_type, 'magnitude')
  images_combined = abs(images_combined);
end
if strcmp(output_image_precision, 'single')
  images_combined = single(images_combined);
  if ~isempty(images)
    images = single(images);
  end
end

end

% =======================================================================
% Kaiser-Bessel kernel  [1,2]
% =======================================================================
function beta = kb_beta(W, osf)
% Optimal Kaiser-Bessel shape parameter for oversampling ratio osf
% (Beatty, Nishimura & Pauly, IEEE TMI 24(6):799-808, 2005 [2]).
arg = (W/osf)^2*(osf-0.5)^2 - 0.8;
if arg <= 0
  beta = pi*sqrt(max((W/osf)^2*0.25, 0.1));
else
  beta = pi*sqrt(arg);
end
end

function v = kb_kernel(u, W, beta)
% Separable 1-D Kaiser-Bessel interpolation kernel [1].
v = zeros(size(u));
t = 2*u/W;
m = abs(t) <= 1;
v(m) = besseli(0, beta*sqrt(1 - t(m).^2));
end

function v = interp_kernel(u, W, beta, ktype, F)
% Separable 1-D interpolation kernel.
%   'kb'   Kaiser-Bessel [1,2] (needs de-apodization afterwards)
%   'sinc' truncated sinc scaled by the FOV-expansion factor F.  The Fourier
%          transform of sinc(u/F) is a rect of width 1/F, i.e. essentially FLAT
%          over the finally cropped FOV, so NO de-apodization is required -
%          this is the classical band-limited-interpolation form of gridding.
if strcmpi(ktype, 'sinc')
  t = u/F;
  v = zeros(size(u));
  m = abs(t) <= 1;
  v(m) = sinc_local(t(m));
else
  v = kb_kernel(u, W, beta);
end
end

function y = sinc_local(x)
% sin(pi x)/(pi x) without needing the Signal Processing Toolbox.
y = ones(size(x));
n = x ~= 0;
y(n) = sin(pi*x(n))./(pi*x(n));
end

function apod1 = kb_deapodization(G, W, beta)
% Analytic Fourier transform of the Kaiser-Bessel kernel [1,2]:
%   c(u) = I0(beta*sqrt(1-(2u/W)^2)),  |u| <= W/2
%   C(x) = W * sinh(sqrt(beta^2-A^2))/sqrt(beta^2-A^2),  A^2 <  beta^2
%   C(x) = W * sin (sqrt(A^2-beta^2))/sqrt(A^2-beta^2),  A^2 >= beta^2
% with A = pi*W*x/G and x the centred image coordinate.
x = (0:G-1).' - floor(G/2);
A = pi*W*x/G;
s = A.^2 - beta^2;
r = sqrt(abs(s));

apod1 = zeros(G, 1);
pos = s > 0;
apod1(pos) = sin(r(pos))./r(pos);
neg = ~pos;
rn = r(neg);
tmp = sinh(rn)./rn;
tmp(rn == 0) = 1;
apod1(neg) = tmp;
apod1 = W*apod1;
end

% =======================================================================
% Sparse interpolation operator
% =======================================================================
function PHI = build_sparse_interpolator(gk, G, W, beta, ktype, F)
% PHI [M x G^3] sparse: row m holds the Kaiser-Bessel weights that spread
% sample m onto its W^3 neighbouring Cartesian grid points (wrapping around,
% as the FFT is periodic).  Assembling it once turns gridding into a single
% sparse matrix product.
M = size(gk, 1);
base = floor(gk);
Wh = floor(W/2);
offs = (-Wh+1):Wh;
nOff = numel(offs)^3;

rows = zeros(M, nOff);
cols = zeros(M, nOff);
vals = zeros(M, nOff);
sample_idx = (1:M).';
c = 0;
for ox = offs
  gx = base(:, 1) + ox;
  wx = interp_kernel(gk(:, 1) - gx, W, beta, ktype, F);
  ix = mod(gx, G);
  for oy = offs
    gy = base(:, 2) + oy;
    wy = interp_kernel(gk(:, 2) - gy, W, beta, ktype, F);
    iy = mod(gy, G);
    for oz = offs
      gz = base(:, 3) + oz;
      wz = interp_kernel(gk(:, 3) - gz, W, beta, ktype, F);
      iz = mod(gz, G);
      c = c + 1;
      rows(:, c) = sample_idx;
      cols(:, c) = ix + G*iy + G*G*iz + 1;
      vals(:, c) = wx.*wy.*wz;
    end
  end
end
keep = vals ~= 0;
PHI = sparse(rows(keep), cols(keep), vals(keep), M, G^3);
end

% =======================================================================
% Density compensation
% =======================================================================
function w = dcf_least_squares(PHI)
% ONE-SHOT LEAST-SQUARES / POINT-SPREAD-FUNCTION-MATCHED density compensation.
%
% Published formulation of gridding as a least-squares problem:
%   D. Rosenfeld, MRM 40(1):14-23, 1998, doi:10.1002/mrm.1910400103       [3]
%   H. Sedarat & D. G. Nishimura, IEEE TMI 19(4):306-317, 2000,
%       doi:10.1109/42.848182  (gridding == least squares)                [4]
%   A. A. Samsonov, E. G. Kholmovski & C. R. Johnson, Proc. ISMRM 2003,
%       p.477 (PSF-modelling gridding DCF - direct prior art)             [5]
%   N. Dwork et al., arXiv:2106.06660, 2021 (LS optimal DCF)              [6]
%   K. O. Johnson & J. G. Pipe, MRM 61(2):439-447, 2009,
%       doi:10.1002/mrm.21840                                             [8]
%
% Let A = PHI*PHI' be the sample-to-sample point-spread operator.  Choosing,
% for every sample m, the scalar weight that makes its point spread function
% closest to a delta in the least-squares sense,
%
%     w_m = argmin_alpha || alpha*A(:,m) - e_m ||^2
%         = <A(:,m), e_m> / ||A(:,m)||^2
%         = A_mm / (A^2)_mm ,
%
% and, using A = A' and A^2 = PHI*(PHI'*PHI)*PHI',
%
%     numerator_m   = (PHI*PHI')_mm            = sum_n |PHI(m,n)|^2
%     denominator_m = (PHI*Gram*PHI')_mm ,  Gram = PHI'*PHI.
%
% Closed form - no iterations, hence much faster than [7].
%
% -----------------------------------------------------------------------
% RELATION TO SEDARAT & NISHIMURA (2000), EQ. (8)  -- PRIOR ART FOR THE
% IDEA OF A ONE-STEP, CLOSED-FORM LEAST-SQUARES DCF
% -----------------------------------------------------------------------
% The notion that the DCF's can be obtained in CLOSED FORM (no iteration)
% by posing gridding as a least-squares approximation problem is NOT new
% here; it is the central result of [4].  In their notation T is the
% L-by-N interpolation matrix (L trajectory samples, N grid points), which
% is exactly our PHI, and D = diag(d_ii) is the diagonal DCF matrix.  They
% write gridding as mhat = T'*D*mu and least-squares reconstruction (LSR)
% as mhat = T'*P*mu with P = T*(T'*T)^(-2)*T', so gridding is LSR with the
% dense compensation matrix P replaced by a DIAGONAL D.  Their Section III
% ("Optimal Density Compensation", p.308) minimises the operator norm of
% the resulting error matrix E = T'*(P - D) and obtains, as Eq. (8),
%
%     d_ii = [ T*(T'*T)^(-1)*T' ]_ii / [ T*T' ]_ii ,     i = 1...L
%
% and their Summary (Section VII) states: "Based on this approach, a
% closed form solution for the optimal DCF's is obtained."  Their Eq. (10)
% further identifies [T*T']_ii as the energy the kernel centred on sample
% i deposits on the grid (>0, ->1 for an infinite grid), guaranteeing the
% ratio is well defined.
%
% The weights computed above are the SAME KIND of closed-form diagonal
% least-squares DCF, differing only in the error functional minimised:
%
%   [4], Eq. (8)  minimise || T'*(P - D) ||  (gridding output ~ LSR output)
%                 -> d_ii = [T (T'T)^-1 T']_ii / [T T']_ii
%   this function minimise || A*D - I ||     (point spread function ~ delta)
%                 -> w_m  = [T T']_mm        / [(T T')^2]_mm ,   A = T*T'
%
% i.e. both are diagonal approximations obtained by a single least-squares
% projection; ours approximates A^(-1) directly (the PSF-modelling variant
% of Samsonov, Kholmovski & Johnson [5], described by [6] as a kernel-based
% method that "consider[s] a finite set of points in the objective
% function", and grouped by [8] with the "algebraic methods [that] deduce a
% solution for the appropriate sampling density weights that minimizes
% error in the gridded k-space").
%
% Eq. (8) of [4] is not used verbatim here because its numerator requires
% (T'*T)^(-1) with T'*T of size N-by-N (N = oversampled grid points, ~10^7
% in 3-D), which is not tractable at this scale, whereas (A^2)_mm needs
% only sparse products with the N-by-N Gram matrix.
Gram = PHI'*PHI;
numer = full(sum(PHI.*conj(PHI), 2));
denom = real(full(sum((PHI*Gram).*conj(PHI), 2)));
w = zeros(size(numer));
ok = denom > 0;
w(ok) = numer(ok)./denom(ok);
w(~isfinite(w)) = 0;
w(w < 0) = 0;
end

function w = dcf_pipe_menon(PHI, n_iter)
% Iterative density compensation, Pipe & Menon, MRM 41(1):179-186, 1999 [7]:
%   w <- w ./ (A*w),   A = PHI*PHI'
% (applied matrix-free as PHI*(PHI'*w) so that A is never formed).
M = size(PHI, 1);
w = ones(M, 1);
for it = 1:n_iter
  den = abs(PHI*(PHI'*w));
  den(den < eps) = eps;
  w = w./den;
end
w = real(w);
w(~isfinite(w)) = 0;
end

function w = dcf_radial_shell(k_ro)
% Analytic density compensation for 3-D radial sampling from the exact volume
% of the spherical shell each sample represents:
%   w(r) = int_{r-D/2}^{r+D/2} r^2 dr = r^2*D + D^3/12
% with D the LOCAL radial sample spacing measured from the trajectory itself
% (so ramp-sampled / centre-out / non-uniform spokes are handled correctly).
% The D^3/12 term gives the correct NON-ZERO weight for the sample at the
% centre of k-space; a plain |k|^2 law wrongly sets it to zero, which discards
% the DC sample of every spoke.
%
%   k_ro : [nReadout x nSpokes x 3];  w : [nReadout*nSpokes x 1]
r = sqrt(sum(k_ro.^2, 3));

n_ro = size(k_ro, 1);
D = zeros(size(r));
if n_ro > 1
  step = sqrt(sum(diff(k_ro, 1, 1).^2, 3));
  D(1, :) = step(1, :);
  D(end, :) = step(end, :);
  if n_ro > 2
    D(2:end-1, :) = 0.5*(step(1:end-1, :) + step(2:end, :));
  end
else
  D(:) = 1;
end

w = (r.^2 + (D.^2)/12).*D;
w = w(:);
w(~isfinite(w)) = 0;
end

% =======================================================================
% Transforms and misc helpers
% =======================================================================
function out = ortho_ifft3(in)
out = in;
for d = 1:3
  out = fftshift(ifft(ifftshift(out, d), [], d), d)*sqrt(size(in, d));
end
end

function img = ifft_crop_deapod(Gk, selected_indices, apod3)
img = ortho_ifft3(Gk);
img = img(selected_indices, selected_indices, selected_indices);
img = img./apod3;
end

function images_combined = allocate_combined_image(final_image_size, num_subsets, adaptive_combine, coil_profiles)
if adaptive_combine > 0 || ~isempty(coil_profiles)
  images_combined = complex(zeros([final_image_size num_subsets]));
else
  images_combined = zeros([final_image_size num_subsets]);
end
end

function out = box_filter3(in, kernel)
% Uses imfilter when the Image Processing Toolbox is available, convn otherwise,
% so this file stays free of toolbox dependencies.
if exist('imfilter', 'file') == 2
  out = imfilter(in, kernel, 'replicate');
else
  out = convn(in, kernel, 'same');
end
end

% =======================================================================
% Minimal name-value option parser
% =======================================================================
function [varargout] = process_options(args, varargin)
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
