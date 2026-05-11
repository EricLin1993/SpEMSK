function  [subsets] = GenerateSubsets_SliWin_lep(WinL,Stride,spokes)
% generate spoke index subsets using a sliding window  

  ne = WinL;
  n = 0;
  while ne <= spokes 
     n = n+1; 
     subsets{n}=ne-WinL+1:ne ;
     ne = ne+Stride;
  end


end
