# mirt_csp_simplify

This folder contains a minimal standalone extraction of the MIRT/IRT coil
sensitivity simulation function used by the package.

Use:

```matlab
addpath(fullfile(packageDir, 'mirt_csp_simplify'), '-begin');
CSP = ir_mri_sensemap_sim('nx', nx, 'ny', ny, 'nz', nz, ...
    'scale', 'ssos_center', 'ncoil', NumCoil, ...
    'orbit_start', -90, 'rcoil', 250);
```

Source:

- Original function: `mirt-main/mri/ir_mri_sensemap_sim.m`
- Original project: Michigan Image Reconstruction Toolbox / IRT
- Original authors: Jeff Fessler and Amanda Funai, University of Michigan
- 3-D modification: Mai Le

License:

This extracted code follows the MIT License used by MIRT/IRT. The full MIT
license notice is included in `ir_mri_sensemap_sim.m`.
