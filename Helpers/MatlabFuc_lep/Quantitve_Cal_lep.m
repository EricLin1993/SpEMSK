function [mean_out,std_out] = Quantitve_Cal_lep(QFun,Image,SliceInd,FrameInd)
% 
 
    if isempty(SliceInd)
       SliceInd = 1:size(Image,3);
    end    
    if isempty(FrameInd)
       FrameInd = 1:size(Image,4);
    end    

    n = 0;
    for nf = FrameInd
     for ns = SliceInd 
      
 
      X = squeeze(Image(ns,:,:,nf)); %  
      n = n+1;   
      q(n) = QFun(X); 

      X = squeeze(Image(:,ns,:,nf)); 
      n = n+1;  
      q(n) = QFun(X); 

      X = squeeze(Image(:,:,ns,nf)); 
      n = n+1;  
      q(n) = QFun(X); 

     end
    end    

    mean_out = mean(q); 
    std_out = std(q); 


end