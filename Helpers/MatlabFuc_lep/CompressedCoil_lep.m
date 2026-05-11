
function [kSpace_Compressed,sv] = CompressedCoil_lep(kSpace,varargin)

        [r,energy,PlotCoef] = process_options_lep(varargin,'r',1,'Energy',[],'PlotCoef',false);        
        sz = size(kSpace);
        kSpace_vec = reshape(kSpace,[],sz(end));
        C = kSpace_vec'*kSpace_vec;

        [v,s] = eig(C);
         
        v = fliplr(v);
        sv = flipud(diag(s) );
 
        if ~isempty(energy) &&  energy>0 && energy <=1 
            s2 = sv.*sv;
             
            s2 = s2/sum(s2);
            r = 0;
            se = 0;
            while se < energy
               r = r+1; 
               se = se+s2(r);
            end    
        end    
        
        P = v(:,1:r);
        kSpace_Compressed = kSpace_vec*P;

        kSpace_Compressed = reshape( kSpace_Compressed,[sz(1:end-1),r]);

          if PlotCoef
            x = 1:length(sv);  % x轴坐标
            Used_Pertentage = sum(sv(1:r))/sum(sv)*100;
            figure
            hold on  
            % 前 r 个为红色
            plot(x(1:r), sv(1:r), 'o', 'MarkerSize', 6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r')
            
            % 之后的为蓝色
            plot(x(r+1:end), sv(r+1:end), 'o', 'MarkerSize', 6,'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b')
            title(['Component Coef (Used:',num2str(Used_Pertentage),'%)']),xlabel('PCA Component'),ylabel('Value'),  
            legend('Used','Rest')
            EnhancePlot_lep;
          end
end












