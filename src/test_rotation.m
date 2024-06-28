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
                                     'density', density);

%% Totation 90 degrees axe 3: x3 or z

C11 = 182.1;
C22 = 280.2;
C33 = 207.6;
C44 =  68.8;
C55 =  56.8;
C66 =  68.5;
C12 =  71.9;
C13 =  70.1;
C23 =  67.2;

% Elastic stiffness tensor (in GPa)
Cij_Ol_90x3 =...
    [[C11   C12   C13   0.0   0.0   0.0];...
    [ C12   C22   C23   0.0   0.0   0.0];...
    [ C13   C23   C33   0.0   0.0   0.0];...
    [ 0.0   0.0   0.0   C44   0.0   0.0];...
    [ 0.0   0.0   0.0   0.0   C55   0.0];...
    [ 0.0   0.0   0.0   0.0   0.0   C66]];

% generate stiffness tensor
tensor_olivine_90x3 = stiffnessTensor(Cij_Ol_90x3,...
                                     'unit', 'GPa',...
                                     cs_olivine,...
                                     'density', density);

%% Totation 90 degrees axe 1: x1 or x?

C11 = 280.2;
C22 = 207.6;
C33 = 182.1;
C44 =  68.8;
C55 =  56.8;
C66 =  68.5;
C12 =  67.2;
C13 =  71.9;
C23 =  70.1;

% Elastic stiffness tensor (in GPa)
Cij_Ol_90x1 =...
    [[C11   C12   C13   0.0   0.0   0.0];...
    [ C12   C22   C23   0.0   0.0   0.0];...
    [ C13   C23   C33   0.0   0.0   0.0];...
    [ 0.0   0.0   0.0   C44   0.0   0.0];...
    [ 0.0   0.0   0.0   0.0   C55   0.0];...
    [ 0.0   0.0   0.0   0.0   0.0   C66]];

% generate stiffness tensor
tensor_olivine_90x1 = stiffnessTensor(Cij_Ol_90x1,...
                                     'unit', 'GPa',...
                                     cs_olivine,...
                                     'density', density);


%% Totation 90 degrees axe 2: x2 or y?

C11 = 207.6;
C22 = 182.1;
C33 = 280.2;
C44 =  68.5;
C55 =  68.8;
C66 =  56.8;
C12 =  70.1;
C13 =  67.2;
C23 =  71.9;

% Elastic stiffness tensor (in GPa)
Cij_Ol_90x2 =...
    [[C11   C12   C13   0.0   0.0   0.0];...
    [ C12   C22   C23   0.0   0.0   0.0];...
    [ C13   C23   C33   0.0   0.0   0.0];...
    [ 0.0   0.0   0.0   C44   0.0   0.0];...
    [ 0.0   0.0   0.0   0.0   C55   0.0];...
    [ 0.0   0.0   0.0   0.0   0.0   C66]];

% generate stiffness tensor
tensor_olivine_90x2 = stiffnessTensor(Cij_Ol_90x2,...
                                     'unit', 'GPa',...
                                     cs_olivine,...
                                     'density', density);

%%
% estimate vp velocities

% Reference
[vp, ~, ~, ~, ~, ~] = velocity(tensor_olivine_ref);
[maxVp, maxVpPos] = max(vp);
[minVp, minVpPos] = min(vp);

% Rotation 90 x3
[vp_x3, ~, ~, ~, ~, ~] = velocity(tensor_olivine_90x3);
[maxVp_x3, maxVpPos_x3] = max(vp_x3);
[minVp_x3, minVpPos_x3] = min(vp_x3);

% Rotation 90 x1
[vp_x1, ~, ~, ~, ~, ~] = velocity(tensor_olivine_90x1);
[maxVp_x1, maxVpPos_x1] = max(vp_x1);
[minVp_x1, minVpPos_x1] = min(vp_x1);

% Rotation 90 x2
[vp_x2, ~, ~, ~, ~, ~] = velocity(tensor_olivine_90x2);
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
%% Totation 45 degrees axe 3: x3 or z

C11 = 220.03;
C22 = 220.03;
C33 = 207.6;
C44 =  62.8;
C55 =  62.8;
C66 =  79.63;
C12 =  83.03;
C13 =  68.65;
C23 =  68.65;
C16 = -24.53;
C26 = -24.53;
C36 =  1.45;
C45 =  -6.0;

% Elastic stiffness tensor (in GPa)
Cij_Ol_45x3 =...
    [[C11   C12   C13   0.0   0.0   C16];...
    [ C12   C22   C23   0.0   0.0   C26];...
    [ C13   C23   C33   0.0   0.0   C36];...
    [ 0.0   0.0   0.0   C44   C45   0.0];...
    [ 0.0   0.0   0.0   C45   C55   0.0];...
    [ C16   C26   C36   0.0   0.0   C66]];

% generate stiffness tensor
tensor_olivine_45x3 = stiffnessTensor(Cij_Ol_45x3,...
                                     'unit', 'GPa',...
                                     cs_olivine,...
                                     'density', density);

%% Totation 45 degrees axe 1: x1 or x?

C11 = 280.2;
C22 = 200.98;
C33 = 200.98;
C44 =  62.8;
C55 =  62.8;
C66 =  62.38;
C12 =  69.55;
C13 =  69.55;
C23 =  63.98;
C16 = -2.35;
C26 =  6.38;
C36 =  6.38;
C45 =  -6.0;

% Elastic stiffness tensor (in GPa)
Cij_Ol_45x1 =...
    [[C11   C12   C13   0.0   0.0   C16];...
    [ C12   C22   C23   0.0   0.0   C26];...
    [ C13   C23   C33   0.0   0.0   C36];...
    [ 0.0   0.0   0.0   C44   C45   0.0];...
    [ 0.0   0.0   0.0   C45   C55   0.0];...
    [ C16   C26   C36   0.0   0.0   C66]];

% generate stiffness tensor
tensor_olivine_45x1 = stiffnessTensor(Cij_Ol_45x1,...
                                     'unit', 'GPa',...
                                     cs_olivine,...
                                     'density', density);


%% Totation 45 degrees axe 2: x2 or y?

C11 = 224.35;
C22 = 182.1;
C33 = 224.35;
C44 =  62.65;
C55 =  88.35;
C66 =  62.65;
C12 =  71.0;
C13 =  86.75;
C23 =  71.0;
C15 = -18.15;
C25 =  -0.9;
C35 =  -18.15;
C46 =  5.85;

% Elastic stiffness tensor (in GPa)
Cij_Ol_45x2 =...
    [[C11   C12   C13   0.0   C15   0.0];...
    [ C12   C22   C23   0.0   C25   0.0];...
    [ C13   C23   C33   0.0   C35   0.0];...
    [ 0.0   0.0   0.0   C44   0.0   C46];...
    [ C15   C25   C35   0.0   C55   0.0];...
    [ 0.0   0.0   0.0   C46   0.0   C66]];

% generate stiffness tensor
tensor_olivine_45x2 = stiffnessTensor(Cij_Ol_45x2,...
                                     'unit', 'GPa',...
                                     cs_olivine,...
                                     'density', density);

%%
% estimate vp velocities

% Rotation 45 x3
[vp_x3, ~, ~, ~, ~, ~] = velocity(tensor_olivine_45x3);
[maxVp_x3, maxVpPos_x3] = max(vp_x3);
[minVp_x3, minVpPos_x3] = min(vp_x3);

% Rotation 45 x1
[vp_x1, ~, ~, ~, ~, ~] = velocity(tensor_olivine_45x1);
[maxVp_x1, maxVpPos_x1] = max(vp_x1);
[minVp_x1, minVpPos_x1] = min(vp_x1);

% Rotation 45 x2
[vp_x2, ~, ~, ~, ~, ~] = velocity(tensor_olivine_45x2);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
rot3 = rotation.byAxisAngle(zvector, 45*degree);
tensor_olivine_rot45zvector = rotate(tensor_olivine_ref, rot3);

rot1 = rotation.byAxisAngle(xvector, 45*degree);
tensor_olivine_rot45xvector = rotate(tensor_olivine_ref, rot1);

rot2 = rotation.byAxisAngle(yvector, 45*degree);
tensor_olivine_rot45yvector = rotate(tensor_olivine_ref, rot2);



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