%%%%%%%%%%%%%%%%%%%%%%Measurement Log%%%%%%%%%%%%%%%%%%%% 
%MeasuredRatio = 2718.36 for PbS, the lower limit of photoresponse peak is
%around 2.473 um. The optimum value is 2.73 um. 

%MeasuredRatio = 254.088 for PbSe, the lower limit of photoresponse peak is
%around 3.602 um. The optimum value is 4.41 um. 

%MeasuredRatio = 640.670 for bulk BP D2, the lower limit of photoresponse peak
%is around  3.075 um. The optimum value is  um.  

%MeasuredRatio = 29.1952 for bulk BP D18, the lower limit of photoresponse peak
%is around  3.507 um. The optimum value is 4.366 um. 

%MeasuredRatio = 33.8846 for bulk BP D19, the lower limit of photoresponse peak
%is around  3.355 um. The optimum value is 4.040 um. 

%MeasuredRatio = 37.6893 for bulk BP D18 repeat, 
%the lower limit of photoresponse peak is around 3.252 um. The optimum value is 4.048 um.

%MeasuredRatio = 43.3725 for bulk BP D19 repeat, 
%the lower limit of photoresponse peak is around 3.124 um. The optimum value is 4.080 um.

%MeasuredRatio = 8.6563 for BP Dual D3.9, 
%the lower limit of photoresponse peak is around 5.568 um. The optimum value is 4.080 um.

%MeasuredRatio = 26.4202 for BP Dual D2.7, 
%the lower limit of photoresponse peak is around 3.612 um. The optimum value is 3.612 um.
%%%%%%%%%%%%%%%%%%%%%%Log End%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Wlen = @(x) (x*1e-6); % convert unit to um for convenience

%import measurement result
TmptureIphm = readtable('t2.5.csv');
Iphm = table2array(TmptureIphm(:,2));


Tstart = 400;
Tend = 800;
Tlength = 21;
T = linspace(Tstart,Tend,Tlength); %Temperature range
Lstart = Wlen(1); % 3.675 or 1
Llength = 21;
Lendmin = Wlen(2);
Lendmax = Wlen(9.5);
Lendnum = 100;
nlambda = 9;
Repeat = 50;
Lend = linspace(Lendmin,Lendmax,Lendnum)';
Mean_mse = zeros(Lendnum,nlambda);
Testpercent = 0.05;

for Lnum  = 1:Lendnum
    disp(Lnum);
    tic;

    L = linspace(Lstart,Lendmax,20); %wavelength range
    Blackbody = @BlackbodyInte; 
    B = Blackbody(T,L,0); % produce blackbody source curve
    %BlackbodyPlot(B,T,L); % plot blackbody curves to check
    %cross validation to minimize mse
    %try nlambda values of lambda and repeat Remo
    Lambda = 10.^-3;
    mse = zeros(Repeat,nlambda); % mse values in nlambda loop
    disp(toc);
    for i = 1:Repeat
       % VMLF = fitrlinear(B,Iphm,'FitBias',false,'KFold',10,'IterationLimit',1e10,'GradientTolerance',1e-20,...
       % 'Regularization','ridge','Lambda',lambda,'Weights',Weight,'BetaTolerance',1e-20,...
       % 'Learner','svm','solver','lbfgs'); 
       % testindex = randi(Tlength,ceil(Testpercent*Tlength),1);
       testindex = [1, 2];
       % testX= B;
       % testY = Iphm;
       testX = B(testindex,:);
       testY = Iphm(testindex,:);
       TrainX = B;
       TrainX(testindex,:) = [];
       TrainY = Iphm;
       TrainY(testindex,:) = [];
       Weight = (1./TrainY).^2;
       [CVMdl,info] = fitrlinear(TrainX,TrainY,'FitBias',false,...
             'Regularization','ridge','Lambda',Lambda,'Weights',Weight,...
             'Learner','leastsquares','solver','bfgs','Iterationlimit',1e4,...
             'OptimizeLearnRate',true ,'GradientTolerance',1e-30,...
             'BetaTolerance',1e-20);
       %CVMdl = fitrlinear(TrainX,TrainY,'FitBias',false,...
        % 'Regularization','ridge','Lambda',Lambda,'Weights',Weight,...
         %'Learner','leastsquares','solver','bfgs');%,...
         %'Iterationlimit',1e6,'OptimizeLearnRate',true,'GradientTolerance',1e-20,...
         %'BetaTolerance',1e-15,'HyperparameterOptimizationOptions',struct('Optimizer','randomsearch')); 
       EstimateY = testX*CVMdl.Beta;
       %WeightY = (1./testY).^2;
       WeightedMse = ((EstimateY-repmat(testY,1,nlambda))).^2;%((EstimateY-repmat(testY,1,nlambda))).^2;%
       mse(i,:) = mean(WeightedMse,1);
       for j = 1:nlambda
           if min(CVMdl.Beta(:,j))< 0
               mse(i,j) = NaN;
           end
       end
    end           
    Mean_mse(Lnum,:) = mean(mse,1);
    disp(toc);
end
logmse = log10(Mean_mse);


