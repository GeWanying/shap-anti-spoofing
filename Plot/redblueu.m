function y = redblueu(varargin)
% original source:
%% https://la.mathworks.com/matlabcentral/fileexchange/74791-redblue-colormap-generator-with-zero-as-white-or-black?s_tid=FX_rc2_behav

% generate a RED-BLUE Uniform colormap with zero as white.
% Syntax: y = redblue(n,clim,'black')
%   Typical usage: colormap(redblue(64))
%   Positive values are displayed as blue intensities and negative
%   values displayed as red intensities. Zero is white.
%   The clim values are used to find zero to set that to white.
% Arguments:
%   All arguments are optional and can be in any order.
%   n - number of color levels 
%       (optional, default is # of colors of current colormap)
%   clim - two element vector specifying the color limits
%       (optional, default: current axis color limits)
%   black - string ('k' or 'black') specifying zero as black.
%       (optional,default is zero as white)
% See: colormap
%
% Notes:
%   This creates a custom colormap for any image so that the value 
%   zero is either white or black. The colorbar scale will be skewed
%   toward red or blue depending on the caxis values of the image.
%
%   Keep in mind that if the scale is very skewed, there will not
%   be much of a color gradient. The gradient can always be increased
%   by using your own clim values.
% Example:
%   y = caxis;                      % e.g. y = [-11,-5] 
%   colormap(redblue(64)            % and not much gradient
%   colormap(redblue(64,[-11,0]))   % white is -5 with larger gradient
% created   3/24/2020   Mirko Hrovat    mihrovat@outlook.com
% modified  5/04/2020   Changed algorithm for colormap calculation, eliminating
%   an error which occurs for small "clim" differences.
n = [];     clim = [];      black = false;
for k = 1:nargin
    a = varargin{k};
    switch true
        case ischar(a)
            switch true
                case strcmpi(a,'k')
                    black = true;
                case strcmpi(a,'black')
                    black = true;
            end
        case isnumeric(a) && numel(a) == 1
            n = a;
        case isnumeric(a) && numel(a) == 2
            clim = a;
    end
end
if isempty(n)
    n = size(colormap,1);
end
if isempty(clim)
    clim = caxis;
end
cmin = min(clim);
cmax = max(clim);
switch true
    case cmin >= 0      % display just blue
        v = linspace(cmin/cmax,1,n);
        y = 1 - repmat(v',[1,3]);
        y(:,3) = 1;
    case cmax <= 0      % display just red
%         v = linspace(1,cmax/cmin,n);
        v = linspace(1,cmax/cmin,n);
        y = 1 - repmat(v',[1,3]);
        y(:,1) = 1;
    otherwise           % display both red and blue
        v = abs(linspace(cmin,cmax,n)/max(abs(cmax),abs(cmin)));
        [~,m] = min(v);
        y = 1 - (repmat(v',[1,3]));
        y(m+1:end,3) = 1;       % set blue
        y(1:m,1) = 1;           % set red
end


if black
    y2 = y;
    y(:,1) = (1-y2(:,2)).*(y2(:,1)==1);
    y(:,3) = (1-y2(:,2)).*(y2(:,3)==1);
    y(:,2) = 0;
end
end