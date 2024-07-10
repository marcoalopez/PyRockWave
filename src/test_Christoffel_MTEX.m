%%
clear variables

%%
Ol_density = 3.343;  % Mao et al. (g/cm3) RP-RT

% Crystal reference frame
cs_olivine = crystalSymmetry('mmm', [4.8 10 6], 'mineral', 'Forsterite', 'color', '#81b29a');

% elastic constants of San Carlos olivine at RP-RT
C11 = 320.2;
C22 = 196.5;
C33 = 232.3;
C44 =  63.2;
C55 =  76.6;
C66 =  79.7;
C12 =  71.0;
C13 =  71.0;
C23 =  76.0;

% Elastic stiffness tensor (in GPa)
Cij =...
    [[C11   C12   C13   0.0   0.0   0.0];...
    [ C12   C22   C23   0.0   0.0   0.0];...
    [ C13   C23   C33   0.0   0.0   0.0];...
    [ 0.0   0.0   0.0   C44   0.0   0.0];...
    [ 0.0   0.0   0.0   0.0   C55   0.0];...
    [ 0.0   0.0   0.0   0.0   0.0   C66]];

tensor_olivine = stiffnessTensor(Cij,...
                                 'unit', 'GPa',...
                                 cs_olivine,...
                                 'density', Ol_density);

%%
tensor_olivine.M

%% set a grid with a resolution of 5 degree
XY_grid = equispacedS2Grid('upper', 'resolution', 1.5*degree);

%% estimate velocities using grid
[vp, vs1, vs2, pp, ps1, ps2] = velocity(tensor_olivine, XY_grid);
% [vp, vs1, vs2, pp, ps1, ps2] = velocity(tensor_olivine, 'harmonic');

%% plot example
plotSeismicVelocities(tensor_olivine)
colormap(flipud(brewermap(256, 'Spectral')))

%%
fprintf('max P-wave velocity = %.4f (km/s)\n', max(vp));
fprintf('min P-wave velocity = %.4f (km/s)\n', min(vp));
disp ' ';
fprintf('max S1-wave velocity = %.4f (km/s)\n', max(vs1));
fprintf('min S1-wave velocity = %.4f (km/s)\n', min(vs1));
disp ' ';
fprintf('max S2-wave velocity = %.4f (km/s)\n', max(vs2));
fprintf('min S2-wave velocity = %.4f (km/s)\n', min(vs2));
disp ' ';


