function [smap, x, y, z] = ir_mri_sensemap_sim(varargin)
%IR_MRI_SENSEMAP_SIM Minimal standalone MIRT coil sensitivity simulator.
%
% This file is a compact extraction of the coil sensitivity simulation path
% from Jeff Fessler's Michigan Image Reconstruction Toolbox (MIRT/IRT).
% It keeps only the code needed to generate 2-D/3-D simulated MRI coil
% sensitivity maps and removes display, test, setup, and path-management
% dependencies from the full toolbox.
%
% Original source:
%   mirt-main/mri/ir_mri_sensemap_sim.m
%   Michigan Image Reconstruction Toolbox / IRT
%
% Original authors:
%   Jeff Fessler and Amanda Funai, University of Michigan
%   3-D modification by Mai Le
%
% Simplified packaging :
%   Enping Lin, Harvard Medical School
%
% License:
%   This extracted file is distributed under the MIT License used by MIRT.
%   See the license notice at the end of this file.
%
% Options:
%   'nx','ny','nz'        Image size. Default [64 64 1].
%   'dx','dy','dz'        Voxel dimensions. Default [3 3 3].
%   'ncoil'              Total number of coils. Default 4.
%   'nring'              Number of coil rings. Default 1.
%   'rcoil'              Coil radius. Default 100.
%   'dz_coil'            Spacing between coil rings.
%   'coil_distance'      Coil-center distance as a multiple of half FOV.
%   'orbit'              Angular span in degrees. Default 360.
%   'orbit_start'        Starting angle in degrees. Default 0.
%   'scale'              '' or 'ssos_center'.
%
% Output:
%   smap                 [nx ny nz ncoil] complex single sensitivity maps.

if nargin < 1
    error('ir_mri_sensemap_sim requires name-value input arguments.');
end

arg.nx = 64;
arg.ny = [];
arg.nz = 1;
arg.dx = 3;
arg.dy = [];
arg.dz = [];
arg.ncoil = 4;
arg.nring = 1;
arg.rcoil = 100;
arg.orbit = 360;
arg.orbit_start = 0;
arg.dz_coil = [];
arg.coil_distance = 1.2;
arg.scale = '';
arg.chat = false;

arg = parse_name_value_options(arg, varargin);

if isempty(arg.dy), arg.dy = arg.dx; end
if isempty(arg.dz), arg.dz = arg.dx; end
if isempty(arg.ny), arg.ny = arg.nx; end
if isempty(arg.rcoil), arg.rcoil = arg.dx * arg.nx / 2 * 0.50; end
if isempty(arg.dz_coil), arg.dz_coil = arg.dz * arg.nz / arg.nring; end

coils_per_ring = round(arg.ncoil / arg.nring);
if arg.nring * coils_per_ring ~= arg.ncoil
    error('nring must be a divisor of ncoil.');
end

[ring_smap, x, y, z] = simulate_sensemap( ...
    arg.nx, arg.ny, arg.nz, ...
    arg.dx, arg.dy, arg.dz, ...
    arg.ncoil, coils_per_ring, arg.rcoil, arg.dz_coil, ...
    arg.orbit, arg.orbit_start, arg.coil_distance);

if arg.nz == 1
    smap = reshape(ring_smap, [arg.nx arg.ny arg.ncoil]);
    scale_center = 1 / sqrt(sum(abs(smap(end/2,end/2,:)).^2));
else
    smap = reshape(ring_smap, [arg.nx arg.ny arg.nz arg.ncoil]);
    scale_center = 1 / sqrt(sum(abs(smap(end/2,end/2,end/2,:)).^2));
end

switch arg.scale
    case ''
    case 'ssos_center'
        smap = smap * scale_center;
    otherwise
        error('Unknown scale method "%s".', arg.scale);
end

end

function [smap, x, y, z] = simulate_sensemap(nx, ny, nz, ...
    dx, dy, dz, ncoil, ncoil_per_ring, rcoil, dz_coil, ...
    orbit, orbit_start, coil_distance)
% Build cylindrical coil geometry and evaluate the circular-coil field.

nring = ncoil / ncoil_per_ring;
rlist = rcoil * ones(ncoil_per_ring, nring, 'single');

plist = zeros(ncoil_per_ring, nring, 3, 'single');
nlist = zeros(ncoil_per_ring, nring, 3, 'single');
olist = zeros(ncoil_per_ring, nring, 3, 'single');

if numel(orbit_start) == 1
    orbit_start = repmat(orbit_start, nring);
end

alist = (pi/180) * orbit * (0:(ncoil_per_ring-1)) / ncoil_per_ring;
z_ring = ((1:nring)-(nring+1)/2) * dz_coil;

for ir = 1:nring
    for ic = 1:ncoil_per_ring
        phi = alist(ic) + (pi/180) * orbit_start(ir);
        radius = max(nx/2 * dx, ny/2 * dy) * coil_distance;
        plist(ic,ir,:) = [radius * [cos(phi) sin(phi)] z_ring(ir)];
        nlist(ic,ir,:) = -[cos(phi) sin(phi) 0*z_ring(ir)];
        olist(ic,ir,:) = [-sin(phi) cos(phi) 0];
    end
end

x = ((1:nx) - (nx+1)/2) * dx;
y = ((1:ny) - (ny+1)/2) * dy;
z = ((1:nz) - (nz+1)/2) * dz;
[xx, yy, zz] = ndgrid(x, y, z);

smap = zeros(nx, ny, nz, ncoil_per_ring, nring, 'single');
for ir = 1:nring
    for ic = 1:ncoil_per_ring
        zr = (xx - plist(ic,ir,1)) .* nlist(ic,ir,1) + ...
             (yy - plist(ic,ir,2)) .* nlist(ic,ir,2) + ...
             (zz - plist(ic,ir,3)) .* nlist(ic,ir,3);
        xr = xx .* nlist(ic,ir,2) - yy .* nlist(ic,ir,1);
        yr = zz - plist(ic,ir,3);

        [sx, ~, sz] = circular_coil_field(xr, yr, zr, rlist(ic,ir));

        bx = sz * nlist(ic,ir,1) + sx * olist(ic,ir,1);
        by = sz * nlist(ic,ir,2) + sx * olist(ic,ir,2);
        smap(:,:,:,ic,ir) = bx + 1i * by;
    end
end

smap = smap * rlist(1) / (2*pi);

end

function [smap_x, smap_y, smap_z] = circular_coil_field(x, y, z, coil_radius)
% Magnetic field of a circular coil, following the MIRT implementation.

x = x ./ coil_radius;
y = y ./ coil_radius;
z = z ./ coil_radius;
r = sqrt(x.^2 + y.^2);

m = 4 * r ./ ((1 + r).^2 + z.^2);
[kval, eval] = ellipke(m);
if is_octave()
    kval = reshape(kval, size(m));
    eval = reshape(eval, size(m));
end

smap_z = 2 * ((1 + r).^2 + z.^2).^(-0.5) .* ...
    (kval + (1 - r.^2 - z.^2) ./ ((1 - r).^2 + z.^2) .* eval);
smap_z = smap_z / coil_radius;

smap_r = 2 * z ./ r .* ((1 + r).^2 + z.^2).^(-0.5) .* ...
    ((1 + r.^2 + z.^2) ./ ((1 - r).^2 + z.^2) .* eval - kval);

near_axis = abs(r) < 1e-6;
smap_r(near_axis) = 3 * pi * z(near_axis) ./ ...
    ((1 + z(near_axis).^2).^2.5) .* r(near_axis);
smap_r = smap_r / coil_radius;

if any(isnan(smap_r(:))) || any(isnan(smap_z(:)))
    error('NaN encountered while simulating coil sensitivity maps.');
end

smap_x = smap_r .* div0(x, r);
smap_y = smap_r .* div0(y, r);

end

function opt = parse_name_value_options(opt, args)
% Minimal name-value parser for the options used by ir_mri_sensemap_sim.

if mod(numel(args), 2) ~= 0
    error('Input options must be name-value pairs.');
end

for ii = 1:2:numel(args)
    name = args{ii};
    value = args{ii+1};

    if ~ischar(name) && ~(isstring(name) && isscalar(name))
        error('Option names must be character vectors or scalar strings.');
    end
    name = char(name);

    if ~isfield(opt, name)
        error('Unknown option name "%s".', name);
    end
    opt.(name) = value;
end

end

function out = div0(num, den)
% Divide safely, returning zero where the denominator is zero.

bad = den == 0;
out = num ./ (den + bad) .* (~bad);

end

function out = is_octave()
% True when running in GNU Octave.

persistent cachedValue
if isempty(cachedValue)
    cachedValue = exist('OCTAVE_VERSION', 'builtin') == 5;
end
out = cachedValue;

end

% MIT License
%
% Copyright (c) 2019 Jeff Fessler
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
