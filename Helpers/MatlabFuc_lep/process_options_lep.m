function varargout = process_options_lep(options, varargin)
%  This function process the option arguments in the outer funtion. For
%  example, in example_fun
%
    % function [a,b,c] = example_fun(arg1,arg2,varargin)
    %      [a,b,c] = process_options_lep(varargin,'a',0,'b',4,'c','lep' )
    % end     
%
% Enping Lin, 2024.6.8
% ====================================================================

    % Initialize the output with the default options
    if ~mod(length(varargin),2) && ~mod(length(options),2)
         varargout = varargin(2:2:end);
    elseif mod(length(varargin),2)
        error('name-value input arguments should be present in pair in process_options_lep')
    else
        stack = dbstack;
    
        % Check if there is a calling function
        if length(stack) > 1
            caller_name = stack(2).name; % The calling function is the second in the stack
            error('The name-value input arguments of %s function should be present in pair ', caller_name);
        else
            error('option arguments in process_options_lep should be present in pair');
        end
    end

    % Update the output with any provided options
    for i = 1:2:length(options)
        option_name = options{i};
        option_value = options{i+1};
        for j = 1:length(varargin)/2
            if strcmpi(varargin{2*j-1}, option_name)
                varargout{j} = option_value;
                break;
            end
        end
    end
end
