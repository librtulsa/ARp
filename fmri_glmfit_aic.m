%
% GLM analysis with the autoregression noise autocorrelation model (AR(p))
% and the AICc order selection
% (This code is the simplified version of fmri_glmfit_arp.m)
% (reference: Luo Q, Misaki M, Mulyana B, Wong C-K, Bodurka J (2018).
%  Optimization of Serial Correlation Correction Methods Based on Autoregressive
%  Model in Fast fMRI. Annual Meeting of ISMRM, Paris)
% (AFNI matlab package required)
% 
% imgfn: image filename (AFNI .BRIK format) 
%        4-D fMRI data [nx, ny, nslices, nframes]
% frrange: time frame range for processing [fr1 fr2]
%          all frames if =[]
% fnout: output filename
% fnreg: task regressor file (text format: one column per one regressor)
% fnnui: nuisance regressor file (e.g., motion parameters)
% contr: contrast ([1xD]), if=[]: all task regressors will be set to 1
% drord: the order of drift signal to be removed (0, 1, or 2)
% numlags: maxium lag used for estimating autocorrelations
%           if = []: use default value = round(10/TR))
% FWHM_COR: FWHM (mm) of the Gaussian filter for smoothing rho values
%           if = []: use default value = 5 mm
%           if = 0: no smoothing
% slcgap: slice gap (mm)
% saverho: save the voxel-wise AR orders and coefficients (=1) or not (=0)
% saveres: save residual data (=1) or not (=0, default)
% varargin{1}: order selection criterion
%              = 1(default): corrected Akaike information criterion (AICc)
%              = 2: Schwartz's Bayesian information criterion (SBC or BIC)
%              = 3: threshold of partial ACF (pACF) (Woolrich 2001)
%
% An example for testing the contrast of two task regressors [A-B]
% fnimg='scan10%retr%rvhr+orig';
% frrange=[13 672];
% fnreg='reg_blk_TR500.1D';
% fnnui='scan10_mot.1D';
% fnout='tmp';
% contr=[1, -1];
% drord=1;
% numlags=[];
% FWHM_COR=0;
% slcgap=1;
% saverho=1;
% saveres=1;
% ordsel=1; % AICc=1; BIC=2; pACF=3
% fmri_glmfit_aic(fnimg, frrange, fnreg, fnnui, fnout, contr, drord, numlags, FWHM_COR, slcgap,saverho,saveres,ordsel);
% 
% -----------------------------------------------------------------------------------------------------------------------------------
%
%           Laureate Institute for Brain Research
%
% 7/2/19    Qingfei Luo  original
%
% -----------------------------------------------------------------------------------------------------------------------------------

function fmri_glmfit_aic(fnimg, frrange, fnreg, fnnui, fnout, contr, drord, numlags, FWHM_COR, slcgap, saverho, saveres, varargin)

tic;
Tcutoff=10; % default cutoff lag time (s)
fwhm_ar=0; % fwhm (mm) for smoothing AR orders

% read data
[err, alldata, imginfo, ErrMessage] = BrikLoad (fnimg);
[numxs, numys, numslices, ntotfr] = size(alldata);
voxsz = imginfo.DELTA; % voxel size (mm)
TR = imginfo.TAXIS_FLOATS(2); % TR (s)
fprintf('Voxel size is [%f %f %f] with slice gap = %f mm\n',voxsz(1),voxsz(2),voxsz(3),slcgap);
fprintf('TR is %f sec\n',TR);
if isempty(frrange)
    ifr1 = 1;
    ifr2 = ntotfr;
else
    ifr1 = frrange(1);  ifr2 = frrange(2);
end
fprintf('%d time frames\n', ifr2-ifr1+1);

% setup regressors and contrast
X = load(fnreg);
[nf, nd] = size(X); 
fprintf('design size is %d frames x %d covariates\n', nf, nd);
if isempty(contr)
    contr = ones(1,nd);
end
% add mean and detrending and nuisance regressors (e.g., head motions) 
c1 = (1:nf)';
c2 = c1.*c1;
switch drord
      case 1
            X = [X c1];
      case 2
            X = [X c1 c2];
end
if ~isempty(fnnui)
    Xnu = load(fnnui);
    X = [X Xnu];
    fprintf('%d nuisance regressors\n', size(Xnu,2));
end
X = X(ifr1:ifr2,:);  % remove some frames
[nfr, ntotreg] = size(X);
Xmean = mean(X); % normalize the regressors
X = X-repmat(Xmean,[nfr,1]);
Xn = sqrt(sum(X.^2));
X = X./repmat(Xn,[nfr,1]);
X = [X,ones(nfr,1)]; % add the mean
ntotreg = ntotreg+1;
contr = [contr, zeros(1,ntotreg-nd)];

% setup parameters
numframes = nfr;
numpix=numxs*numys;
if isempty(numlags)
    numlags = round(Tcutoff/TR);
    fprintf('the max lag for the AC estimation is set to %d\n',numlags);
else
    if numlags<1
        fprintf('AR order must be larger than 0');
        return;
    end
end

% ---------------------------------------------
% voxel-wise order selection
% ---------------------------------------------
% calculate autocovariances of fitting residuals (Y-Xb)
n = numframes;
q = size(X,2); % number of regressors
resid_all = zeros(numpix,numslices,numframes); % fitting residuals
Cov=zeros(numpix,numlags+1); % autocovariances of residuals in one slice at lags=[0:numlags]
Cov_all=zeros(numpix,numslices,numlags+1); % autocovariances at lag=0:numlags
Cov0_all=zeros(numpix,numslices); % autocovariance at lag=0
rho_vol_res=squeeze(zeros(numpix, numslices, numlags)); % autocorrelations of residuals
Y=zeros(n,numpix);
pinvX=pinv(X);
emode = 0; % estimate mode: =0:raw; =1:tuckey windowed
if emode==1
    T=numlags+1;
    tau = 1:T-1;
    sfcov = 0.5*(1+cos(tau*pi/T)); % scale factor of covariance
end
for slice=1:numslices
      img = alldata(:,:,slice,ifr1:ifr2);
      Y = reshape(img,numpix,n);  % slice time series data
      Y = Y';     
      betahat_ls=pinvX*Y; % Least squares: 
      resid=Y-X*betahat_ls; % fitting residuals
      resid_all(:,slice,:)=resid';
      for lag=0:numlags  % raw estimate of covariance
            Cov(:,lag+1)=(sum(resid(1:(n-lag),:).*resid((lag+1):n,:)))'/(n-lag);
      end
      switch emode       
             case 1  % Tukey window (Woolrich 2001)
                    Cov(:,1)=Cov(:,1);
                    Cov(:,2:T) =Cov(:,2:T).*repmat(sfcov,[numpix,1]);
      end
      Cov_all(:,slice,:)=Cov;
      Cov0_all(:,slice)=Cov(:,1);
      rho_vol_res(:,slice,:)= Cov(:,2:numlags+1)./repmat(Cov(:,1),[1,numlags]);
end

% select AR(p) model orders with the GLM fitting residuals
if isempty(varargin)
    selmode = 1;
else
    selmode = varargin{1};
end
switch selmode
    case 1
        fprintf('select AR order with AICc\n');
    case 2
        fprintf('select AR order with SBC\n');
    case 3
        fprintf('select AR order with PACF\n');
end
arord_vol=zeros(numpix,numslices); % selected AR orders
th_pac = 2/sqrt(numframes); % threshold of patial autocorrelation
N=numframes;
m=1:numlags;
parfor slice=1:numslices
%     for slice=1:numslices
          % ---------------- NOTE ---------------------------
         % the coefficients a in aryule is -Q in the AR model
         % in aryule: sum(a_k*y_n-k) = x_n (k=0:p)
         % a_0 is always 1, so y_n = -sum(a_k*y_n-k) + x_n (k=1:p)
         % in AR model: y_n = sum(Q_k*y_n-k) + x_n (k=1:p)
         % thus: Q_k = -a_k
         % ---------------------------------------------------

        fprintf('AR fitting slice %d\n',slice);
        E = zeros(numpix,numlags); % variance of white noise
        RC = zeros(numpix,numlags); % partial autocorrelations
        rho = squeeze(rho_vol_res(:,slice,:));
        Cov0=Cov0_all(:,slice); % autocovariance at lag = 0
        for p=1:numlags  % AR model fitting with different orders
%             resid=squeeze(resid_all(:,slice,:)); % calculate variance of AR fitting residuals with Burg's method             
%             [tmpa,tmpe,tmprc]=arburg(resid',p);

            tmprho=rho(:,1:p)'; % calculate the variance of AR fitting with Levionson's method
            [tmpa,tmpe,tmprc] = levinson([ones(1,numpix); tmprho]);
            tmpa=-tmpa(:,2:end);
            tmpe=Cov0.*(1-diag(tmpa*tmprho)); % variance of fitting residuals
            
%             tmprho=rho(:,1:p)'; % use this if the signal processing toolbox is unavailable
%             [tmpa,tmpe]=my_yule(2,[ones(1,numpix); tmprho].*repmat(Cov0',p+1,1),p);  
%             tmpe=tmpe';
%             tmprc=zeros(numpix,p);
            
            tmpe(isnan(tmpe))=0;
            E(:,p)=tmpe;  % variance of white noise [pix,order]
            if p==numlags
                RC=-tmprc; % save the partical ACF [order,pix]
            end
        end
        for i=1:numpix    % select AR orders in voxel-wise
             switch selmode
                 case 1   % Akaike information criterion: AIC=n[log(V_p)+1]+2(p+1)
                     % corrected AIC (AICc): AICc=n*log(V_p)+n(n+p)/(n-p-2)
                     % V_p is the white noise variance at AR order=p
%                        AIC = N.*(log(E(i,:))+1)+2*(m+1); % AIC
                       AIC = N.*log(E(i,:))+N.*(N+m)./(N-m-2); % AICc
                       optAIC=find(AIC==min(AIC));
                       arord_vol(i,slice)=optAIC(1);
%                        arord=optAIC;
                       
                 case 2  % Schwartz's Bayesian criterion (SBC): SBC = N*log(e)+p*log(N)
                       SBC = N.*log(E(i,:))+m.*log(N);
                       optSBC=find(SBC==min(SBC));
                       arord_vol(i,slice)=optSBC(1);
%                        arord=optSBC;
                       
                 case 3 % threshold of PACF
                       if abs(RC(1,i))>0   % determine the AR order (Woolrich 2001)
                            p = find(abs(RC(:,i))<th_pac);
                            if isempty(p)
                                arord_vol(i,slice)=numlags;
                            else
                                arord_vol(i,slice) = p(1);
                            end
                       else 
                            arord_vol(i,slice) = 0;
                       end
%                        arord=arord_vol(i,slice);           
             end
        end % end of loop pixel
end
minord = min(arord_vol(:));
maxord = max(arord_vol(:));
fprintf('the minimum/maximum raw AR order is = %d/%d\n',minord,maxord);

% smooth AR orders
if fwhm_ar>0
    arord_vol = reshape(arord_vol,numxs,numys,numslices);
    arordsm = arord_vol; % smoothed orders
    Step=voxsz;
    Step(3)=Step(3)+slcgap;
    sigma = fwhm_ar./(2.35*Step);
    arordsm = round(imgaussfilt3(arordsm,sigma));
    arord_vol=reshape(arordsm,numpix,numslices);
    minord = min(arord_vol(:));
    maxord = max(arord_vol(:));
    fprintf('the minimum/maximum smoothed AR order is = %d/%d\n',minord,maxord);
end

% --------------------------------------------------------
% Estimate autocorrelation function (ACF) of noise
% --------------------------------------------------------
% estimate autocovariances of residuals using the AR(p) model
% and power spectrum density method (Friston 2000)
nfft=2^(nextpow2(numframes)+1); 
lag1 = floor(nfft/2)+1; % lag 0
lagn = floor(nfft/2)+numlags+1; % lag numlags
Cov_all1 = zeros(numpix,numslices,numlags+1); % autocovariances
parfor slice = 1:numslices
    for i=1:numpix
        % calculate rho from the PSD (Friston 2000)
%         [tmpa,tmpe,tmprc] = levinson(Covadj(:,i),arord);
%         spec = fft(tmpa',nfft);
%         spec = 1./(abs(fftshift(spec)).^2);   % power spectral density
%         acf = abs(fftshift(fft(spec,nfft),1)); % autocorrelation function
%         acf = acf./max(acf);
        
        % this MATLAB's function is faster and outputs the same result as above
          res = squeeze(resid_all(i,slice,:));
          arord = arord_vol(i,slice);
          pxx = pburg(res',arord,nfft,'centered'); % power spectral density
          acf = abs(fftshift(fft(pxx,nfft)));
          acf = acf./repmat(max(acf),nfft,1); % ACF
          acf=acf(lag1:lagn);
          Cov_all1(i,slice,:)=acf*Cov0_all(i,slice);
    end
end
Cov_all = Cov_all1;

% reduce the bias of autocovariances estimation because the autocovariance
% of noise is not equal to that of fitting residual (V != RVR) (Worsley 2002)
rho_vol=squeeze(zeros(numpix, numslices, numlags)); % ACF of noise
keep = 1:n; % n=numframes
indk1=((keep(2:n)-keep(1:n-1))==1);
k1=find(indk1)+1;   % k1 = 2:numframes
Diag1=diag(indk1,1)+diag(indk1,-1);
R=eye(n)-X*pinvX; % R = R'
if numlags==1
   M11=trace(R);
   M12=trace(R*Diag1);
   M21=M12/2;
   M22=trace(R*Diag1*R*Diag1)/2;
   M=[M11 M12; M21 M22];
else
   M=zeros(numlags+1);
   for i=1:(numlags+1)
       for j=1:(numlags+1)
           Di=(diag(ones(1,n-i+1),i-1)+diag(ones(1,n-i+1),-i+1))/(1+(i==1));
           Dj=(diag(ones(1,n-j+1),j-1)+diag(ones(1,n-j+1),-j+1))/(1+(j==1));
           M(i,j)=trace(R*Di*R*Dj)/(1+(i>1));
       end
   end
end
invM=inv(M);
for slice=1:numslices
      Cov0=Cov0_all(:,slice);
      if numlags==1
         Cov1=squeeze(Cov_all(:,slice,1));
         Covadj=invM*[Cov0; Cov1];
         rho_vol(:,slice)=(Covadj(2,:)./ ...
            (Covadj(1,:)+(Covadj(1,:)<=0)).*(Covadj(1,:)>0))';
      else
         Cov=squeeze(Cov_all(:,slice,:));
         Cov=Cov';
         Covadj=invM*Cov;
         rho_vol(:,slice,:)= ( Covadj(2:(numlags+1),:) ...
                 .*( ones(numlags,1)*((Covadj(1,:)>0)./ ...
                 (Covadj(1,:)+(Covadj(1,:)<=0)))) )';
      end   
end

% smoothing ACF with Gaussian filter
if isempty(FWHM_COR)  % Gaussian filter
    fwhm=5;
else
    fwhm=FWHM_COR;
end
if fwhm>0
    Step=voxsz;
    Step(3)=Step(3)+slcgap;
    sigma = fwhm./(2.35*Step);
    for lag=1:numlags
        tmp=squeeze(rho_vol(:,:,lag));
        tmp=reshape(tmp,numxs,numys,numslices);
        tmp = imgaussfilt3(tmp,sigma);
        rho_vol(:,:,lag)=reshape(tmp,numpix,numslices,1);
    end
end

% -----------------------------------------------------
% loop over voxels to do prewhitening and statistics
% -----------------------------------------------------
res_vol = zeros(numpix,numslices); % standard error of voxel noise
T_vol = zeros(numpix, numslices);
allbeta_vol = zeros(numpix, numslices,q);
Df=(n-rank(X)); % degrees of freedom
parfor slice=1:numslices
% for slice=1:numslices
   fprintf('prewhitening slice %d\n',slice);
   Xstar=X;
   img = alldata(:,:,slice,ifr1:ifr2);
   Y = reshape(img,numpix,n);
   Y = Y';
   Ystar=zeros(n,1);
        
   % loop voxels in the current slice
   for pix = 1:numpix
        Coradj_pix=squeeze(rho_vol(pix,slice,:));
        tmp=find(Coradj_pix==0);
        if ~isempty(tmp)
            Coradj_pix(tmp(1):end)=[];
        end
        Rpix = toeplitz([1 Coradj_pix']);
        [Ainvt posdef]=chol(Rpix);
        nl=size(Ainvt,1);
        A=inv(Ainvt');
        B=ones(n-nl,1)*A(nl,:);
        Vmhalf=spdiags(B,1:nl,n-nl,n);
        Ystar(1:nl)=A*Y(1:nl,pix);
        Ystar((nl+1):n)=Vmhalf*Y(:,pix);
        Xstar(1:nl,:)=A*X(1:nl,:);
        Xstar((nl+1):n,:)=Vmhalf*X;
        
        pinvXstar=pinv(Xstar);
        betahat=pinvXstar*Ystar;
        resid=Ystar-Xstar*betahat;
        allbeta_vol(pix,slice,:) = betahat;
        SSE=sum(resid.^2,1);
        VAR = SSE/Df;
        sd=sqrt(VAR);
        V=contr*(pinvXstar*pinvXstar')*contr'; % keff
        sdbetahat=sqrt(V)*sd;
        E = contr*betahat;
        T_vol(pix,slice)=E./(sdbetahat+(sdbetahat<=0)).*(sdbetahat>0);
        res_vol(pix,slice) = sqrt(VAR);
        resid_all(pix,slice,:) = resid; % save residuals
   end  % end voxel loop
   
end  % end slice loop

%close parallel computation pool
poolobj = gcp('nocreate');
delete(poolobj);

% in case of infinite numbers
T_vol(isnan(T_vol))=0;
T_vol(isinf(T_vol))=0;

%
% save results
%
Opt.master = fnimg;
Opt.datum = 'float';
Opt.OverWrite = 'y';
% write out the t-map and beta-map file
Opt.prefix = [fnout '%t'];
[err,newInfo, newOpt] = New_HEAD(Opt); % create header structure
tmp = zeros(numxs,numys,numslices,ntotreg+2);
tmp(:,:,:,1) = reshape(T_vol,[numxs,numys,numslices,1]);
% tmp(:,:,:,2) = reshape(beta_vol,[numxs,numys,numslices,1]);
tmp(:,:,:,2:end-1) = reshape(allbeta_vol,[numxs,numys,numslices,ntotreg]);
tmp(:,:,:,end) = reshape(res_vol,[numxs,numys,numslices,1]);
str = 't';
for i=1:ntotreg
    tmpstr = ['~beta' num2str(i)];
    str = [str tmpstr];
end
str = [str '~StDev'];
newInfo.BRICK_LABS = str;
[err, errmsg, info] = WriteBrik(tmp,newInfo,newOpt);
% write out other variables
if saverho
    Opt.prefix = [fnout '%rho'];
    [err,newInfo, newOpt] = New_HEAD(Opt);
    tmp = zeros(numxs,numys,numslices,numlags+1);
    tmp(:,:,:,1) = reshape(arord_vol,[numxs,numys,numslices,1]);
    tmp(:,:,:,2:end) = reshape(rho_vol,[numxs,numys,numslices,numlags]);
    str = 'ARord';
    for i=1:numlags
        tmpstr = ['~rho' num2str(i)];
        str = [str, tmpstr];
    end
    newInfo.BRICK_LABS = str;
    [err, errmsg, info] = WriteBrik(tmp,newInfo,newOpt);
end
% write out residual images
if saveres
    Opt.prefix = [fnout '%res'];
    [err,newInfo, newOpt] = New_HEAD(Opt);
    tmp = reshape(resid_all,[numxs,numys,numslices,numframes]);
    [err, errmsg, info] = WriteBrik(tmp,newInfo,newOpt);
end
fprintf('\ndone!\n');
toc;
