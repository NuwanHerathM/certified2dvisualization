polynome
resolution
h = figure('Visible', 'off');
fimplicit(p, 'MeshDensity', n, 'XRange', [-1 1], 'YRange', [-1 1])
print(h, '-dpng', 'image.png')

% timeit(f)
% f = chebfun2(p)
% tic
% r = roots(f)
% plot(r), axis([-1 1 -1 1])
% toc
% contour(f,[0 0])
