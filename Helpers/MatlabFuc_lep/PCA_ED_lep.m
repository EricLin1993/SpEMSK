function [EWinCompress,d,v ]= PCA_ED_lep(EWin,varargin)
  
%  Component == 0: all components

    [Component] = process_options_lep(varargin,'Component',0);
    EhE = EWin'*EWin;
    [v,d] = eig(EhE);
    [d,od] = sort(diag(d),'descend');
    v = v(:,od);
    if Component == 0
         EWinCompress = (v.'*EWin.').';
    else     
        EWinCompress = (v(:,Component).'*EWin.').';
    end    
    

end