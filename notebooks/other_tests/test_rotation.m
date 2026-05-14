%%
clear variables

%%
density = 3.291

% Crystal reference frame
cs_olivine = crystalSymmetry('mmm', [4.8 10 6], 'mineral', 'Forsterite', 'color', '#81b29a');

% elastic constants of San Carlos olivine at 1.5 GPa and 1027°C (1300 K), custom fitting from Zhang and Bass (2016)
C11 = 280.2;
C22 = 182.1;
C33 = 207.6;
C44 =  56.8;
C55 =  68.8;
C66 =  68.5;
C12 =  71.9;
C13 =  67.2;
C23 =  70.1;

% Elastic stiffness tensor (in GPa)
Cij_Ol =...
    [[C11   C12   C13   0.0   0.0   0.0];...
    [ C12   C22   C23   0.0   0.0   0.0];...
    [ C13   C23   C33   0.0   0.0   0.0];...
    [ 0.0   0.0   0.0   C44   0.0   0.0];...
    [ 0.0   0.0   0.0   0.0   C55   0.0];...
    [ 0.0   0.0   0.0   0.0   0.0   C66]];

% generate stiffness tensor
tensor_olivine_ref = stiffnessTensor(Cij_Ol,...
                                     'unit', 'GPa',...
                                     cs_olivine,...
                                     'density', density)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
rot3 = rotation.byAxisAngle(zvector, 90*degree);
tensor_olivine_rot90zvector = rotate(tensor_olivine_ref, rot3)

rot1 = rotation.byAxisAngle(xvector, 90*degree);
tensor_olivine_rot90xvector = rotate(tensor_olivine_ref, rot1)

rot2 = rotation.byAxisAngle(yvector, 90*degree);
tensor_olivine_rot90yvector = rotate(tensor_olivine_ref, rot2)

%%
% estimate vp velocities

% Reference
[vp, ~, ~, ~, ~, ~] = velocity(tensor_olivine_ref);
[maxVp, maxVpPos] = max(vp);
[minVp, minVpPos] = min(vp);

% Rotation 90 x3
[vp_x3, ~, ~, ~, ~, ~] = velocity(tensor_olivine_rot90zvector);
[maxVp_x3, maxVpPos_x3] = max(vp_x3);
[minVp_x3, minVpPos_x3] = min(vp_x3);

% Rotation 90 x1
[vp_x1, ~, ~, ~, ~, ~] = velocity(tensor_olivine_rot90xvector);
[maxVp_x1, maxVpPos_x1] = max(vp_x1);
[minVp_x1, minVpPos_x1] = min(vp_x1);

% Rotation 90 x2
[vp_x2, ~, ~, ~, ~, ~] = velocity(tensor_olivine_rot90yvector);
[maxVp_x2, maxVpPos_x2] = max(vp_x2);
[minVp_x2, minVpPos_x2] = min(vp_x2);


%% figure Vp comparative
blackMarker = {'Marker','s','MarkerSize',14,'antipodal',...
  'MarkerEdgeColor','white','MarkerFaceColor','black','doNotDraw'};
whiteMarker = {'Marker','o','MarkerSize',14,'antipodal',...
  'MarkerEdgeColor','black','MarkerFaceColor','white','doNotDraw'};

mtexFig = newMtexFigure('figSize', 'huge', 'layout', [1 4]);

plot(vp, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine reference', 'doNotDraw')
hold on
plot(maxVpPos(1), blackMarker{:})
hold on
plot(minVpPos(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp = 200*(maxVp - minVp)./(maxVp + minVp);
xlabel({['Anisotropy = ', num2str(AVp,'%6.1f')]; ['min ', num2str(minVp,'%6.1f'), ' - ', num2str(maxVp,'%6.1f'), ' max']})


nextAxis(1,2)
plot(vp_x3, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine 90 degrees z(x3)', 'doNotDraw')
hold on
plot(maxVpPos_x3(1), blackMarker{:})
hold on
plot(minVpPos_x3(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp_x3 = 200*(maxVp_x3 - minVp_x3)./(maxVp_x3 + minVp_x3);
xlabel({['Anisotropy = ', num2str(AVp_x3,'%6.1f')]; ['min ', num2str(minVp_x3,'%6.1f'), ' - ', num2str(maxVp_x3,'%6.1f'), ' max']})


nextAxis(1,3)
plot(vp_x1, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine 90 degrees x(x1)', 'doNotDraw')
hold on
plot(maxVpPos_x1(1), blackMarker{:})
hold on
plot(minVpPos_x1(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp_x1 = 200*(maxVp_x1 - minVp_x1)./(maxVp_x1 + minVp_x1);
xlabel({['Anisotropy = ', num2str(AVp_x1,'%6.1f')]; ['min ', num2str(minVp_x1,'%6.1f'), ' - ', num2str(maxVp_x1,'%6.1f'), ' max']})

nextAxis(1,4)
plot(vp_x2, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine 90 degrees y(x2)', 'doNotDraw')
hold on
plot(maxVpPos_x2(1), blackMarker{:})
hold on
plot(minVpPos_x2(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp_x2 = 200*(maxVp_x2 - minVp_x2)./(maxVp_x2 + minVp_x2);
xlabel({['Anisotropy = ', num2str(AVp_x2,'%6.1f')]; ['min ', num2str(minVp_x2,'%6.1f'), ' - ', num2str(maxVp_x2,'%6.1f'), ' max']})

setColorRange('equal')
mtexColorbar('figSize', 'huge', 'title', 'km/s', 'FontSize', 24)
colormap(flipud(brewermap(256, 'Spectral')))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
rot3 = rotation.byAxisAngle(zvector, 45*degree);
tensor_olivine_rot45zvector = rotate(tensor_olivine_ref, rot3)

rot1 = rotation.byAxisAngle(xvector, 45*degree);
tensor_olivine_rot45xvector = rotate(tensor_olivine_ref, rot1)

rot2 = rotation.byAxisAngle(yvector, 45*degree);
tensor_olivine_rot45yvector = rotate(tensor_olivine_ref, rot2)



%%
% Rotation 45 x3
[vp_x3, ~, ~, ~, ~, ~] = velocity(tensor_olivine_rot45zvector);
[maxVp_x3, maxVpPos_x3] = max(vp_x3);
[minVp_x3, minVpPos_x3] = min(vp_x3);

% Rotation 45 x1
[vp_x1, ~, ~, ~, ~, ~] = velocity(tensor_olivine_rot45xvector);
[maxVp_x1, maxVpPos_x1] = max(vp_x1);
[minVp_x1, minVpPos_x1] = min(vp_x1);

% Rotation 45 x2
[vp_x2, ~, ~, ~, ~, ~] = velocity(tensor_olivine_rot45yvector);
[maxVp_x2, maxVpPos_x2] = max(vp_x2);
[minVp_x2, minVpPos_x2] = min(vp_x2);


%% figure Vp comparative
blackMarker = {'Marker','s','MarkerSize',14,'antipodal',...
  'MarkerEdgeColor','white','MarkerFaceColor','black','doNotDraw'};
whiteMarker = {'Marker','o','MarkerSize',14,'antipodal',...
  'MarkerEdgeColor','black','MarkerFaceColor','white','doNotDraw'};

mtexFig = newMtexFigure('figSize', 'huge', 'layout', [1 4]);

plot(vp, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine reference', 'doNotDraw')
hold on
plot(maxVpPos(1), blackMarker{:})
hold on
plot(minVpPos(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp = 200*(maxVp - minVp)./(maxVp + minVp);
xlabel({['Anisotropy = ', num2str(AVp,'%6.1f')]; ['min ', num2str(minVp,'%6.1f'), ' - ', num2str(maxVp,'%6.1f'), ' max']})


nextAxis(1,2)
plot(vp_x3, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine 45 degrees z(x3)', 'doNotDraw')
hold on
plot(maxVpPos_x3(1), blackMarker{:})
hold on
plot(minVpPos_x3(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp_x3 = 200*(maxVp_x3 - minVp_x3)./(maxVp_x3 + minVp_x3);
xlabel({['Anisotropy = ', num2str(AVp_x3,'%6.1f')]; ['min ', num2str(minVp_x3,'%6.1f'), ' - ', num2str(maxVp_x3,'%6.1f'), ' max']})


nextAxis(1,3)
plot(vp_x1, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine 45 degrees x(x1)', 'doNotDraw')
hold on
plot(maxVpPos_x1(1), blackMarker{:})
hold on
plot(minVpPos_x1(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp_x1 = 200*(maxVp_x1 - minVp_x1)./(maxVp_x1 + minVp_x1);
xlabel({['Anisotropy = ', num2str(AVp_x1,'%6.1f')]; ['min ', num2str(minVp_x1,'%6.1f'), ' - ', num2str(maxVp_x1,'%6.1f'), ' max']})

nextAxis(1,4)
plot(vp_x2, 'contourf', 'complete', 'doNotDraw', 'upper')
mtexTitle('Olivine 45 degrees y(x2)', 'doNotDraw')
hold on
plot(maxVpPos_x2(1), blackMarker{:})
hold on
plot(minVpPos_x2(1), whiteMarker{:})
hold off
% percentage anisotropy
AVp_x2 = 200*(maxVp_x2 - minVp_x2)./(maxVp_x2 + minVp_x2);
xlabel({['Anisotropy = ', num2str(AVp_x2,'%6.1f')]; ['min ', num2str(minVp_x2,'%6.1f'), ' - ', num2str(maxVp_x2,'%6.1f'), ' max']})

setColorRange('equal')
mtexColorbar('figSize', 'huge', 'title', 'km/s', 'FontSize', 24)
colormap(flipud(brewermap(256, 'Spectral')))