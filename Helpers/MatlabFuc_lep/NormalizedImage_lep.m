function [NormalizedImage] = NormalizedImage_lep(Image,varargin )
% 
%  Enping Lin, 20240603

   % if nargin<2
   %   mode = 'max';
   % end   

    [mode] = process_options_lep(varargin,'mode','max' );
        
   switch mode 
       case {'max','MAX','Max'}
          NormalizedImage =Image./max( abs(Image),[],"all");
       case {'sum','SUM','Sum'}
          NormalizedImage = Image./sum( abs(Image),"all");
       case {'norm','NORM','Norm'}
          NormalizedImage = Image./norm( Image(:) ); 
       otherwise
          error('Invalid Mode in NormalizedImage_lep') 
   end
end