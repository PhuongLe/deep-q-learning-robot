% Load initial matrix storing the look-up table with format: energy, distance, gun heat,
% action, Q-value, number of visits
% Offline processing steps: 1. import lut.log file in Matlab
%                           2. convert it into a matrix in the import data
%                           tool
%                           3. rename the matrix accordingly and save it
%                           4. load the saved matrix
%                           5. after processing, rename processed LuT
%                           accordingly and change extension of file from
%                           .txt to .log
load('initialLuTMat.mat');

energyLevels = 3;
distanceLevels = 3;
gunHeatLevels = 2;
actionLevels = 4;

% Process state-action vectors
energyThMat = convertToThCode(energyLevels);
distanceThMat = convertToThCode(distanceLevels);
gunHeatThMat = convertToThCode(gunHeatLevels);
actionThMat = convertToThCode(actionLevels);

% Find maximum value of Q for normalization
[valMax, ~] = max(abs(initialLuTMat(:,5)));
valMaxAbs = abs(valMax);

% Process the table
processedLuT = zeros(size(initialLuTMat,1), energyLevels-1 + distanceLevels-1 + gunHeatLevels-1 + actionLevels-1 + 2);
for iiRow = 1:size(processedLuT,1)
    % Process energy
    energyVal = initialLuTMat(iiRow,1);
    if energyVal == 0
        processedLuT(iiRow, 1:energyLevels-1) = energyThMat(1,:);
    else
        if energyVal == 1
            processedLuT(iiRow, 1:energyLevels-1) = energyThMat(2,:);
        else
            if energyVal == 2
                processedLuT(iiRow, 1:energyLevels-1) = energyThMat(3,:);
            end
        end
    end
    
    % Process distance
    distanceVal = initialLuTMat(iiRow,2);
    distIndex = energyLevels;
    if distanceVal == 0
        processedLuT(iiRow, distIndex:distIndex + (distanceLevels-1) - 1) = distanceThMat(1,:);
    else
        if distanceVal == 1
            processedLuT(iiRow, distIndex:distIndex + (distanceLevels-1) - 1) = distanceThMat(2,:);
        else
            if distanceVal == 2
                processedLuT(iiRow, distIndex:distIndex + (distanceLevels-1) - 1) = distanceThMat(3,:);
            end
        end
    end
    
    % Process gun heat
    gunHeatVal = initialLuTMat(iiRow,3);
    gunHeatIndex = energyLevels + (distanceLevels-1);
    if gunHeatVal == 0
        processedLuT(iiRow, gunHeatIndex:gunHeatIndex + (gunHeatLevels-1) - 1) = gunHeatThMat(1,:);
    else
        if gunHeatVal == 1
            processedLuT(iiRow, gunHeatIndex:gunHeatIndex + (gunHeatLevels-1) - 1) = gunHeatThMat(2,:);
        end
    end
    
    % Process action
    actionVal = initialLuTMat(iiRow,4);
    actionIndex = energyLevels + (distanceLevels - 1) + (gunHeatLevels-1);
    if actionVal == 0
        processedLuT(iiRow, actionIndex:actionIndex + (actionLevels-1) - 1) = actionThMat(1,:);
    else
        if actionVal == 1
            processedLuT(iiRow, actionIndex:actionIndex + (actionLevels-1) - 1) = actionThMat(2,:);
        else
            if actionVal == 2
                processedLuT(iiRow, actionIndex:actionIndex + (actionLevels-1) - 1) = actionThMat(3,:);
            else
                if actionVal == 3
                    processedLuT(iiRow, actionIndex:actionIndex + (actionLevels-1) - 1) = actionThMat(4,:);
                end
            end
        end
    end
    
    % Process Q-value
    processedLuT(iiRow, actionIndex + (actionLevels-1)) = initialLuTMat(iiRow, 5) / valMaxAbs;
    % Store visits
    processedLuT(iiRow, end) = initialLuTMat(iiRow, end);

end

% Save processed LuT matrix
save('processedLuT.mat','processedLuT');

% Write matrix to .txt file
% writematrix(processedLuT, fid, 'Delimiter', 'comma');

% --- 
fid = fopen('2020-12-04-00-57-05-robocode-lut_processed.txt', 'wt');
% Print number of state-action vectors on first line
fprintf(fid, '%d\n', size(processedLuT,1));
% Print number of dimensions on second line
fprintf(fid, '%d\n', size(processedLuT,2) - 2);
% Print processed look-up table
for iiRow = 1:size(processedLuT,1)
  fprintf(fid, '%d,%d,%d,%d,%d,%d,%d,%d,%f,%d\n', processedLuT(iiRow,:));
end
fclose(fid);
