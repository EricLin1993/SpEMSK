function [kSpace_Reordering,traj_Reordering,score_Reordering] = Reordering_Spoke_lep(kSpace,traj,score,WinLen,varargin)
% 
 


        [Order] = process_options_lep(varargin,'Order','ascend');    



        [score_Reordering,AscOrder]=sort( score,Order);
        IndOrder = ceil(WinLen/2):ceil(WinLen/2)+length(score_Reordering)-1;
        kSpace_Reordering = kSpace;
        temp = kSpace_Reordering(:,IndOrder,:);
        kSpace_Reordering(:,IndOrder,:) = temp(:,AscOrder,:);
        traj_Reordering = traj ;
        temp = traj(:,IndOrder,:);
        traj_Reordering(:,IndOrder,:) = temp(:,AscOrder,:);





end