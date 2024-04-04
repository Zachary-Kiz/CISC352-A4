function a4_00000000
% Function for CISC271, Winter 2022, Assignment #4

    % Read the test data from a CSV file
    dmrisk = csvread('dmrisk.csv',1,0);
    new = load('hw10.txt');
    data = new(:,1:2);
    label = new(:,3);
    % Columns for the data and labels; DM is diabetes, OB is obesity
    jDM = 17;
    jOB = 2;

    % Extract the data matrices and labels
    XDM = dmrisk(:, (1:size(dmrisk,2))~=jDM);
    yDM = dmrisk(:,jDM);
    XOB = dmrisk(:, (1:size(dmrisk,2))~=jOB);
    yOB = dmrisk(:,jOB);

    % Reduce the dimensionality to 2D using PCA
    [~,rDM] = pca(zscore(XDM), 'NumComponents', 2);
    [~,rOB] = pca(zscore(XOB), 'NumComponents', 2);

    % Find the LDA vectors and scores for each data set
    [qDM zDM qOB zOB] = a4q1(data, label, rOB, yOB);
    qDM
    zDM
    % %
    % % STUDENT CODE GOES HERE: PLOT RELEVANT DATA
    % %
    figure(1)
    gscatter(zDM, ones(size(rDM, 1), 1).*yDM, yDM)
    axis([-5, 5, -3, 3])
    hold on

    figure(2)
    gscatter(zOB, ones(size(rOB,1),1).*yDM,yDM)
    axis([-5,5,-3,3])
    hold on

    % Compute the ROC curve and its AUC where: "xroc" is the horizontal
    % axis of false positive rates; "yroc" is the vertical
    % axis of true positive rates; "auc" is the area under curve
    % %
    % % STUDENT CODE GOES HERE: COMPUTE, PLOT, DISPLAY RELEVANT DATA
    % %
    [xroc1,yroc1,auc1] = roccurve(yDM, zDM);
    [xroc2,yroc2,auc2] = roccurve(yOB, zOB);
    disp('AUC Diabetes')
    disp(auc1)
    disp('AUC Obesity')
    disp(auc2)
    figure(3)
    plot(xroc1,yroc1)
    figure(4)
    plot(xroc2,yroc2)

% END OF FUNCTION
end

function [q1, z1, q2, z2] = a4q1(Xmat1, yvec1, Xmat2, yvec2)
% [Q1 Z1 Q2 Z2]=A4Q1(X1,Y1,X2,Y2) computes an LDA axis and a
% score vector for X1 with Y1, and for X2 with Y2.
%
% INPUTS:
%         X1 - MxN data, M observations of N variables
%         Y1 - Mx1 labels, +/- computed as ==/~= 1
%         X2 - MxN data, M observations of N variables
%         Y2 - Mx1 labels, +/- computed as ==/~= 1
% OUTPUTS:
%         Q1 - Nx1 vector, LDA axis of data set #1
%         Z1 - Mx1 vector, scores of data set #1
%         Q2 - Nx1 vector, LDA axis of data set #2
%         Z2 - Mx1 vector, scores of data set #2

    q1 = [];
    z1 = [];
    q2 = [];
    z2 = [];
    
    % Compute the LDA axis for each data set
    q1 = lda2class(Xmat1(yvec1==1,:), Xmat1(yvec1~=1, :));
    q2 = lda2class(Xmat2(yvec2==1,:), Xmat2(yvec2~=1, :));
   
    % %
    % % STUDENT CODE GOES HERE: COMPUTE SCORES USING LDA AXES
    % %
    meanX1 = mean(Xmat1(yvec1==1,:)) + mean(Xmat1(yvec1~=1, :));
    meanX2 = mean(Xmat2(yvec2==1,:)) + mean(Xmat2(yvec2~=1, :));
    z1 = (Xmat1 - ones(size(Xmat1,1),1)*meanX1)*q1;
    z2 = (Xmat2 - ones(size(Xmat2,1),1)*meanX2)*q2;
% END OF FUNCTION
end

function qvec = lda2class(X1, X2)
% QVEC=LDA2(X1,X2) finds Fisher's linear discriminant axis QVEC
% for data in X1 and X2.  The data are assumed to be sufficiently
% independent that the within-label scatter matrix is full rank.
%
% INPUTS:
%         X1   - M1xN data with M1 observations of N variables
%         X2   - M2xN data with M2 observations of N variables
% OUTPUTS:
%         qvec - Nx1 unit direction of maximum separation

    qvec = ones(size(X1,2), 1);
    xbar1 = mean(X1);
    xbar2 = mean(X2);

    % Compute the within-class means and scatter matrices
    % %
    % % STUDENT CODE GOES HERE: COMPUTE S1, S2, Sw
    % %
    M1 = X1 - ones(size(X1,1),1)* xbar1;
    M2 = X2 - ones(size(X2,1),1)*xbar2;
    S1 = transpose(M1) * M1;
    S2 = transpose(M2) * M2;
    Sw = S1 + S2;
    % Compute the between-class scatter matrix
    % %
    % % STUDENT CODE GOES HERE: COMPUTE Sb
    % %
    meanX = xbar1 + xbar2;
    data = [xbar1 - meanX;xbar2 - meanX];
    Sb = transpose(data) * data;
    % Fisher's linear discriminant is the largest eigenvector
    % of the Rayleigh quotient
    % %
    % % STUDENT CODE GOES HERE: COMPUTE qvec
    % %
    [eigVecs,eigVals] = eig(inv(Sw)*Sb, "vector");
    eigVals = sort(eigVals);
    eigVecs = sort(eigVecs);
    qvec = eigVecs(:,1);
    % May need to correct the sign of qvec to point towards mean of X1
    if (xbar1 - xbar2)*qvec < 0
        qvec = -qvec;
    end
% END OF FUNCTION
end

function [fpr tpr auc bopt] = roccurve(yvec_in,zvec_in)
% [FPR TPR AUC BOPT]=ROCCURVE(YVEC,ZVEC) computes the
% ROC curve and related values for labels YVEC and scores ZVEC.
% Unique scores are used as thresholds for binary classification.
%
% INPUTS:
%         YVEC - Mx1 labels, +/- computed as ==/~= 1
%         ZVEC - Mx1 scores, real numbers
% OUTPUTS:
%         FPR  - Kx1 vector of False Positive Rate values
%         TPR  - Kx1 vector of  True Positive Rate values
%         AUC  - scalar, Area Under Curve of ROC determined by TPR and FPR
%         BOPT - scalar, optimal threshold for accuracy

    % Sort the scores and permute the labels accordingly
    [zvec zndx] = sort(zvec_in);
    yvec = yvec_in(zndx);
        
    % Sort and find a unique subset of the scores; problem size
    bvec = unique(zvec);
    bm = numel(bvec);
    
    % Compute a confusion matrix for each unique threshold value;
    % extract normalized entries into TPR and FPR vectors; track
    % the accuracy and optimal B threshold
    tpr = [];
    fpr = [];
    acc = -inf;
    bopt = -inf;
    matopt = [];
    for jx = 1:1
        % %
        % % STUDENT CODE GOES HERE: FIND TPR, FPR, OPTIMAL THRESHOLD
        % %
        cmat = confmat(yvec, zvec, 0);
        TP = cmat(1,1);
        P = TP + cmat(1,2);
        FP = cmat(2,1);
        N = cmat(2,1) + cmat(2,2);
        tpr = [tpr; TP / P];
        fpr = [fpr; FP / N];
        accNew = (TP + cmat(2,2)) / (P + N);
        if accNew > acc
            acc = accNew;
            bopt = bvec(jx);
            matopt = cmat;
        end
    end
    acc
    disp('optimal confusion matrix')
    disp(matopt)
    
    % Ensure that the rates, from these scores, will plot correctly
    tpr = sort(tpr);
    fpr = sort(fpr);
    
    % Compute AUC for this ROC
    auc = aucofroc(fpr, tpr);
end
    
function cmat = confmat(yvec, zvec, theta)
% CMAT=CONFMAT(YVEC,ZVEC,THETA) finds the confusion matrix CMAT for labels
% YVEC from scores ZVEC and a threshold THETA. YVEC is assumed to be +1/-1
% and each entry of ZVEC is scored as -1 if <THETA and +1 otherwise. CMAT
% is returned as [TP FN ; FP TN]
%
% INPUTS:
%         YVEC  - Mx1 values, +/- computed as ==/~= 1
%         ZVEC  - Mx1 scores, real numbers
%         THETA - threshold real-valued scalar
% OUTPUTS:
%         CMAT  - 2x2 confusion matrix; rows are +/- labels,
%                 columns are +/- classifications

    % Find the plus/minus 1 vector of quantizations
    qvec = sign((zvec >= theta) - 0.5);
    
    % Compute the confusion matrix by entries
    % %
    % % STUDENT CODE GOES HERE: COMPUTE MATRIX
    % %
    TP = 0;
    FN = 0;
    FP = 0;
    TN = 0;
    for idx = 1:numel(yvec)
        if yvec(idx) == 1 && qvec(idx) == 1
            TP = TP + 1;
        end
        if yvec(idx) ~= 1 && qvec(idx) ~= 1
            TN = TN + 1;
        end
        if yvec(idx) == 1 && qvec(idx) ~= 1
            FN = FN + 1;
        end
        if yvec(idx) ~= 1 && qvec(idx) == 1
            FP = FP + 1;
        end
        
    end
    cmat = [TP FN; FP TN];
end

function auc = aucofroc(fpr, tpr)
% AUC=AUCOFROC(TPR,FPR) finds the Area Under Curve of the
% ROC curve specified by the TPR, True Positive Rate, and
% the FPR, False Positive Rate.
%
% INPUTS:
%         TPR - Kx1 vector, rate for underlying score threshold 
%         FPR - Kx1 vector, rate for underlying score threshold 
% OUTPUTS:
%         AUC - integral, from Trapezoidal Rule on [0,0] to [1,1]

    [X undx] = sort(reshape(fpr, 1, numel(fpr)));
    Y = sort(reshape(tpr(undx), 1, numel(undx)));
    auc = abs(trapz([0 X 1] , [0 Y 1]));
end
