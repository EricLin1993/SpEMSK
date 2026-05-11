function [ y] = Scale_lep( x,range)
% 
  
   if range(2)<range(1)
      error('range(2) should be greater than range(1)!')
   end    

   x = x-min(x(:));
   x = x/max(x(:)) *(range(2)-range(1)) ;
   y = x+range(1);
   

end