function [SE_score] = Extra_SpE_PosInf_lep(kSpace,WinLen,PC)
% 


    
    [EWin,xx] = kSpokeEnergy_SldWin_lep(kSpace,WinLen,'Scaled',true,'display_yes',false);
    [SEWinCompress,d ]= PCA_ED_lep( EWin,'Component',PC);     
    temp = SEWinCompress-min(SEWinCompress(:));
    SE_score = temp/max(temp(:));

 

end