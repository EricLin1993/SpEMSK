function [X] = ChangeView_lep(X,plane)
%  


if iscell(X)

    for it = 1:length(X)   
       switch plane
      
       case {'axial','Axial','A','a','AXIAL','tranverse','Tranverse','T','TRANVERSE','t'}
              
       case {'saggital','Saggital','S','SAGGITAL','s'}    
          X{it} = permute(X{it} ,[3,1,2,4]);
          X{it}  = flip(X{it},1);
       case {'coronal','Coronal','C','CORONAL','c'}
          X{it}  = permute(X{it} ,[3,2,1,4]);
          X{it}  = flip(X{it} ,1);
       end
    end    

elseif isnumeric(X)

   switch plane
      
       case {'axial','Axial','A','a','AXIAL','tranverse','Tranverse','T','TRANVERSE','t'}
              
       case {'saggital','Saggital','S','SAGGITAL','s'}    
          X = permute(X,[3,1,2,4]);
          X = flip(X,1);
       case {'coronal','Coronal','C','CORONAL','c'}
          X = permute(X,[3,2,1,4]);
          X = flip(X,1);
   end

else

    error('Input variable is of wrong type')

end




end