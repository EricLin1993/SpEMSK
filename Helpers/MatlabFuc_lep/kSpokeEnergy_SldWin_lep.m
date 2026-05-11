
function [EWin,xx] = kSpokeEnergy_SldWin_lep(kSpace,WinLen,varargin)




    [ Zero_Offest , Scaled ,display_yes ] = process_options_lep(varargin,'Zero_Offest',false,'Scaled',false,'display_yes',true);
    
    ReadoutN = size(kSpace,1);
     
    kSpace_temp = reshape(kSpace,ReadoutN,1,[],size(kSpace,3));

    E =  squeeze(sum(conj(kSpace_temp).*kSpace_temp,[1,2]));
    EWin = zeros(size(E,1)-WinLen+1,size(E,2));      
    kernel = ones(WinLen,1);
    for it = 1:size(E,2)
       EWin(:,it) = conv(E(:,it),kernel,'valid'); % 'same'
    end    
    if Zero_Offest
        EWin = EWin-EWin(1,:) ;
    end
    if Scaled
        temp = EWin-min(EWin(:));
        EWin = temp/max(temp(:));  
    end    
    xx = WinLen:WinLen+size(EWin,1)-1;
    
    if display_yes
        figure,plot(xx,EWin,'linewidth',1.5), xlabel('spoke')
        xlim([WinLen,WinLen+size(EWin,1)-1])
        if Scaled
            ylim([-0.1,1.1]);  
        end  

    end
    % title(['Sliding Window of ',num2str(WinLen),' spokes']);
    EnhancePlot_lep; 
     
    


end













