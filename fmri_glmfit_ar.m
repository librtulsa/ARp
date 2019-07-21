%
% GLM analysis with the autoregression noise autocorrelation model (AR(p))
% (reference: Worsley et al. (2002) NeuroImage 15:1-15)
% (This code is modified from fmrilm.m from Keith Worsley's FMRISTAT toolbox)
% (AFNI matlab package required)
% 
% imgfn: functional image filename (AFNI .BRIK format) 
%        4-D fMRI data [nx, ny, nslices, nframes]
% frrange: time frame range for processing [fr1 fr2]
%          all frames if =[]
% fnout: output filename, if =[]: no writeout
% fnreg: task regressor file (text format: one column per regressor)
% fnnui: nuisance regressor file (e.g., motion parameters)
% contr: contrast ([1xD]), if=[]: all task regressors will be set to 1
% drord: the order of drift signal to be removed (0, 1, or 2)
% numlags: AR order number (p) in the AR model (AR(p))
%           if = []: use default value = 1   (AR(1) model)
% FWHM_COR: FWHM Gaussian kernel size (mm) for smoothing rho in the AR model
%           if = []: use default value = 15 mm
%           if = 0: no smoothing
% slcgap: slice gap (mm)
% saverho: save the AR coefficients
% saveres: save residual images
% varargin{1}: voxel-wise AR order, default: all voxels = numlags
% varargout{1}: residuals after prewhitening and GLM fit
% varargout{2}: autocorrelations [nx*ny, nslices, nlags]
%
% An example for testing the contrast of two task regressors [A-B] with AR(5):
% fnimg='scan10%retr%rvhr+orig';
% frrange=[13 672];
% fnreg='reg_blk_TR500.1D';
% fnnui='scan10_mot.1D';
% fnout='tmp';
% contr=[1, -1];
% drord=1;
% numlags=5;
% FWHM_COR=0;
% slcgap=1;
% saverho=1;
% saveres=1;
% fmri_glmfit_ar(fnimg, frrange, fnreg, fnnui, fnout, contr, drord, numlags, FWHM_COR, slcgap,saverho,saveres);
%
% ---------------------------------------------------------------------------------------------------------------
%
%           Laureate Institute for Brain Research
%
% 9/19/17   Qingfei Luo  original
% 7/2/19    Qingfei Luo  clean version
%
% ---------------------------------------------------------------------------------------------------------------

function [varargout] = fmri_glmfit_ar(fnimg, frrange, fnreg, fnnui, fnout, contr, drord, numlags, FWHM_COR, slcgap,saverho,saveres,varargin)

tic;
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

% smoothing filter size
if isempty(FWHM_COR)
    FWHM_COR = 15;
    fprintf('set FWHM of smoothing filter to 15 mm\n');
end

% setup parameters
numframes = nfr;
numpix=numxs*numys;
Steps=voxsz;
Steps(3)=Steps(3)+slcgap;
n = numframes;
q = size(X,2);
if isempty(numlags)
    numlags = 1;
else
    if numlags<1
        fprintf('AR order must be larger than 0');
        return;
    end
end

% Setup for finding rho:
rho_vol=squeeze(zeros(numpix, numslices, numlags));
res_vol_all = zeros(numpix, numslices,numframes); % residuals after GLM fit
T_vol = zeros(numpix, numslices);
allbeta_vol = zeros(numpix, numslices,q);
res_vol = zeros(numpix,numslices);
keep = 1:n;
indk1=((keep(2:n)-keep(1:n-1))==1);
k1=find(indk1)+1;

% First loop over slices, then pixels, to get the AR parameter:
Diag1=diag(indk1,1)+diag(indk1,-1);
Y=zeros(n,numpix);
pinvX=pinv(X);
dfs=n-rank(X); % degree of freedom
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

% Preliminary calculations for unbiased estimates of autocorrelation
for slice=1:numslices
      img = alldata(:,:,slice,ifr1:ifr2);
      Y = reshape(img,numpix,n);  % slice time series data
      Y = Y';     
      % Least squares: 
      betahat_ls=pinvX*Y;
      resid=Y-X*betahat_ls;
      if numlags==1
         Cov0=sum(resid.*resid,1);
         Cov1=sum(resid(k1,:).*resid(k1-1,:),1);
         Covadj=invM*[Cov0; Cov1];
         rho_vol(:,slice)=(Covadj(2,:)./ ...
            (Covadj(1,:)+(Covadj(1,:)<=0)).*(Covadj(1,:)>0))';
      else
         for lag=0:numlags
              Cov(lag+1,:)=sum(resid(1:(n-lag),:).*resid((lag+1):n,:))/(n-lag);
         end
         Covadj=invM*Cov;
         Cov0=Cov(1,:); 
         rho_vol(:,slice,:)= ( Covadj(2:(numlags+1),:) ...
                 .*( ones(numlags,1)*((Covadj(1,:)>0)./ ...
                 (Covadj(1,:)+(Covadj(1,:)<=0)))) )';
      end 
      
end

% smoothing rho with Gaussian filter
fwhm_cor=FWHM_COR;
if fwhm_cor>0
    sigma = fwhm_cor./(2.35*Steps);
    for lag=1:numlags
        tmp=squeeze(rho_vol(:,:,lag));
        tmp=reshape(tmp,numxs,numys,numslices);
        tmp = imgaussfilt3(tmp,sigma);
        rho_vol(:,:,lag)=reshape(tmp,numpix,numslices,1);
    end
end

% cutoff lags
if ~isempty(varargin)
    tmpord=varargin{1};
    tmpord=reshape(tmpord,numpix,numslices);
    for i=1:numpix
        for j=1:numslices
            id=tmpord(i,j);
            if id<numlags
                rho_vol(i,j,id+1:end)=0;
            end
        end
    end
end

% loop over voxels to get statistics
% start parallel computation pool
parfor slice=1:numslices
% for slice=1:numslices
   fprintf('slice %d\n',slice);
   Df=dfs;
   img = alldata(:,:,slice,ifr1:ifr2);
   Y = reshape(img,numpix,n);
   Y = Y';
   % loop voxels in the current slice
   for pix = 1:numpix
      Ystar=Y(:,pix);
      Xstar=X;
      if numlags==1
         rho = rho_vol(pix,slice);
         factor=1./sqrt(1-rho^2);
         Ystar(k1,:)=(Y(k1,pix)-rho*Y(k1-1,pix))*factor;
         Xstar(k1,:)=(X(k1,:)-rho*X(k1-1,:))*factor;
      else
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
         Ystar=zeros(n,1);
         Ystar(1:nl)=A*Y(1:nl,pix);
         Ystar((nl+1):n)=Vmhalf*Y(:,pix);
         Xstar(1:nl,:)=A*X(1:nl,:);
         Xstar((nl+1):n,:)=Vmhalf*X;          
      end
      pinvXstar=pinv(Xstar);
      betahat=pinvXstar*Ystar;
      resid=Ystar-Xstar*betahat;
      allbeta_vol(pix,slice,:) = betahat;
      SSE=sum(resid.^2,1);
      VAR = SSE/Df; % variance
      sd=sqrt(VAR); % standard deviation
      V=contr*(pinvXstar*pinvXstar')*contr';
      sdbetahat=sqrt(diag(V))*sd;
      E = contr*betahat;
      T_vol(pix,slice)=E./(sdbetahat+(sdbetahat<=0)).*(sdbetahat>0);
      res_vol(pix,slice) = sd;
      res_vol_all(pix,slice,:)=resid;
   end  % end voxel loop
   
end  % end slice loop
%close parallel computation pool
poolobj = gcp('nocreate');
delete(poolobj);

% in case of infinite numbers
T_vol(isnan(T_vol))=0;
T_vol(isinf(T_vol))=0;

%
% output results
%
varargout{1}=res_vol_all;
varargout{2}=rho_vol;
if ~isempty(fnout)
Opt.master = fnimg;
Opt.datum = 'float';
Opt.OverWrite = 'y';
% write out the t-map and beta-map file
Opt.prefix = [fnout '%t'];
[err,newInfo, newOpt] = New_HEAD(Opt); % create header structure
tmp = zeros(numxs,numys,numslices,ntotreg+2);
tmp(:,:,:,1) = reshape(T_vol,[numxs,numys,numslices,1]);
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
    tmp = reshape(rho_vol,[numxs,numys,numslices,numlags]);
    str = 'rho';
    for i=1:numlags-1
        tmpstr = [num2str(i) '~rho'];
        str = [str tmpstr];
    end
    str = [str num2str(numlags)];
    newInfo.BRICK_LABS = str;
    [err, errmsg, info] = WriteBrik(tmp,newInfo,newOpt);
end
    % write out residual images
    if saveres
        Opt.prefix = [fnout '%res'];
        [err,newInfo, newOpt] = New_HEAD(Opt);
        tmp = reshape(res_vol_all,[numxs,numys,numslices,nfr]);
        [err, errmsg, info] = WriteBrik(tmp,newInfo,newOpt);
    end
end
fprintf('\ndone!\n');
toc;
