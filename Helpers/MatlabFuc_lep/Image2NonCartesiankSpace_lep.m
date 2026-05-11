function [kSpace_Mo,traj ]= Image2NonCartesiankSpace_lep(ImageCombined,CSP,traj,varargin)

% Author: Enping Lin
% Date: 2026.5.7


[Interp_Method,subset] = process_options_lep(varargin,'Interp_Method','bilinear','subset',{1:size(traj,2)});



kSpace_Mo = zeros(size(traj,1),size(traj,2),size(CSP,4));
N = size(ImageCombined,1);
co = linspace(-0.5, 0.5-1/N, N); %
[X, Y, Z] = ndgrid(co,co,co); 
sn = length(subset);

vec = @(x) x(:);
 

for it0 = 1:sn

    ImageCoil = ImageCombined(:,:,:,it0).*CSP;
    for it1 = 1:size(ImageCoil,4) 
        image_zs = ImageCoil(:,:,:,it1);
        image_zs =  fftshift(image_zs) ;
        kSpace_cart_s = fftn( image_zs )./sqrt(prod(size(image_zs))  );
        kSpace_cart_s =  fftshift(kSpace_cart_s) ;
        temp = interpn(X, Y, Z, kSpace_cart_s, vec(traj(:,subset{it0},1)), vec(traj(:,subset{it0},2)), vec(traj(:,subset{it0},3)),Interp_Method);  
        kSpace_Mo(:,subset{it0},it1) =reshape(temp,size(traj,1),[]);
    end
end   

kSpace_Mo(isnan(kSpace_Mo)) = 0;
% traj_Mo = traj(:,cell2mat(subset),:);

% img_SpE_Seg = tf_nufft_radial_recon(kSpace_Mo(:,1:1:4480,:), traj(:,1:1:4480,:), 128, 210, 'down_sampling_factor', 1);  %data.trajectory  'subsets',{Subsets_SE{1}} 
% imxshow_lep_v2(img_SpE_Seg,'plane','c' )

end






