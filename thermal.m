function XN=NeiX(X)
[s,t,K]=size(X);
Xu1=zeros(s,t,K);
Xu1(2:s,2:t,:)=X(1:s-1,1:t-1,:);

Xu=zeros(s,t,K);
Xu(2:s,:,:)=X(1:s-1,:,:);

Xur=zeros(s,t,K);
Xur(2:s,1:t-1,:)=X(1:s-1,2:t,:);

Xr=zeros(s,t,K);
Xr(:,1:t-1,:)=X(:,2:t,:);

Xdr=zeros(s,t,K);
Xdr(1:s-1,1:t-1,:)=X(2:s,2:t,:);

Xd=zeros(s,t,K);
Xd(1:s-1,:,:)=X(2:s,:,:);

Xd1=zeros(s,t,K);
Xd1(1:s-1,2:t,:)=X(2:s,1:t-1,:);

X1=zeros(s,t,K);
X1(:,2:t,:)=X(:,1:t-1,:);
end

function [X, R] = imstack2vectors(S, MASK)
[M, N, n] = size(S);
if nargin == 1
   MASK = true(M, N);
else
   MASK = MASK ~= 0;
end

[I, J] = find(MASK);
R = [I, J];

Q = M*N;
X = reshape(S, Q, n);

MASK = reshape(MASK, Q, 1);

X = X(MASK, :);


function [E]=EnergyOfLabelField(segmentation,potential,width,height,class_number)
n=size(segmentation,1);
segmentation=reshape(segmentation,[width height]);
Nei8=imstack2vectors(NeiX(segmentation));
E=zeros(n,class_number);
for i=1:class_number
    E(:,i)=sum(Nei8~=i,2);
end
E=E*potential;
end

function[mu,sigma]=GMM_parameter(image,segmentation,class_number)
[n,d]=size(image);
mu=zeros(class_number,d);
sigma=zeros(d,d,class_number);
   for i=1:class_number
       Im_i=image(segmentation==i,:);
       [sigma(:,:,i),mu(i,:)]=covmatrix(Im_i);
    end
end

function segmentation=ICM(image,class_number,potential,maxIter)
[width,height,bands]=size(image);
image=imstack2vectors(image);
[segmentation,c]=kmeans(image,class_number);
%segmentation=reshape(id,[width height]);
clear c;
iter=0;
while(iter<maxIter)
    [mu,sigma]=GMM_parameter(image,segmentation,class_number);
    Ef=EnergyOfFeatureField(image,mu,sigma,class_number);
    E1=EnergyOfLabelField(segmentation,potential,width,height,class_number);
    E=Ef+E1;
    [tm,segmentation]=min(E,[],2);
    iter=iter+1;
end
segmentation=reshape(segmentation,[width height]);
end

function [E]=EnergyOfFeatureField(image,mu,sigma,class_number)
n=size(image,1);
E=zeros(n,class_number);
for i=1:class_number
    mu_i=mu(i,:);
    sigma_i=sigma(:,:,i);
    diff_i=image-repmat(mu_i,[n,1]);
    E(:,i)=sum(diff_i*inv(sigma_i).*diff_i,2)+log(det(sigma_i));
end
end

function [C,m]=covmatrix(X)
[K,n]=size(X);
X=double(X);
if K==1
    C=0;
    m=X;
else
    m=sum(X,1)/K;
    X=X-m(ones(K,1),:);
    C=(X'*X)/(K-1);
    m=m';
end
end

clc
clear

I=imread('If Led Zeppelin Goes Down, We All Burn.jpg');
I=double(I);
class_number=3;
potential=0.5;
maxIter=30;
seg=ICM(I,class_number,potential,maxIter);
figure;
imshow(I);