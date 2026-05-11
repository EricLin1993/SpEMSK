function  EnhancePlot_lep(varargin)
%
    [LW,FS] = process_options_lep(varargin,'LineWidth',2,'FontSize',12);

    axes_handles = findall(gcf, 'Type', 'axes');
    for i = 1:length(axes_handles)
        set(axes_handles(i), 'LineWidth', LW);  
        set(axes_handles(i), 'FontWeight', 'bold');  
        set(axes_handles(i), 'FontSize', FS);  
    end

end