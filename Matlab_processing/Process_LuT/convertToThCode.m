function [thCodeMat] = convertToThCode(numLevels)

numBits = numLevels - 1; % number of bits needed for the thermometer code
initVec = repmat([-1], 1, numBits);
thCodeMat = repmat(initVec, numLevels, 1);

for iLev = 1:numLevels
    % Select level
    selLevel = thCodeMat(iLev, :);
    if iLev == 1
        % Keep initial level
        thCodeMat(iLev, :) = selLevel;
    else
        selLevel(:, end : -1 : numLevels-iLev+1) = ones(1, iLev - 1);
        thCodeMat(iLev, :) = selLevel;
    end
end

end