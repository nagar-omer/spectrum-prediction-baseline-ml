IphD = Imea(:,1);
smooth(:,1);
length = size(IphD,1);
Repeat = 100;
nlambda = 9;
Lambda = 10.^linspace(-2,2,nlambda);
mse = zeros(Repeat,nlambda);
Testpercent = 0.05;
Rex = Rex5um81interp';%Rex21interp';
Negmark = zeros(nlambda,1);
%Rex(isnan(Rex))=0;

for i = 1:Repeat
        %VMLF = fitrlinear(B,Iphm,'FitBias',false,'KFold',10,'IterationLimit',1e10,'GradientTolerance',1e-20,...
       % 'Regularization','ridge','Lambda',lambda,'Weights',Weight,'BetaTolerance',1e-20,...
       % 'Learner','svm','solver','lbfgs'); 
       testindex = randi(length,ceil(Testpercent*length),1);
       testX = Rex(testindex,:);
       testY = IphD(testindex,:);
       TrainX = Rex;
       TrainX(testindex,:) = [];
       TrainY = IphD;
       TrainY(testindex,:) = [];
       Weight = (1./TrainY).^2;
       
       Specfit = fitrlinear(TrainX,TrainY,'FitBias',false,...'Beta',0.4235*initial,...
         'Regularization','ridge','Lambda',Lambda,'Weights',Weight,...
         'Learner','leastsquares',...'Solver','asgd',...
         'Iterationlimit',1e4,...
         'OptimizeLearnRate',true ,...'GradientTolerance',1e-30,...
         'BetaTolerance',1e-20); 
     
       EstimateY = testX*Specfit.Beta;
       RawMSE = (EstimateY-repmat(testY,1,nlambda)).^2;
       mse(i,:) = mean(RawMSE,1);
       for j = 1:nlambda
           if min(Specfit.Beta(:,j))< 0
               Negmark(j,1) = 1;
           end
       end
end
Mean_mse = mean(mse,1);
figure;
plot(log10(Lambda),log10(Mean_mse));


