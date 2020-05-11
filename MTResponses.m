clear;  
%% FORWARD FORMULATION

f = [1000 700 500 300 200 150 100 70 50 30 20 15 10 7 5 3 2 1.5 1 0.7 0.5 0.3 0.2 0.15 0.1 0.07 0.05 0.03 0.02 0.015 0.01 0.007 0.005 0.003 0.002 0.0015 0.001];
om = 2*pi*f;
%rho = [1000 20 5000 50];
%h = [500 2000 5000];
rho = [80 8 80 8];                      %change parameters for resistivity
h = [1000 5000 10000];                  %change parameters for thickness of layer
mu = 4*pi*10^(-7);
K = sqrt(j*om*mu);
n = length(rho);
for i = 1:length(f)
    Z(n) = K(i)*sqrt(rho(n));
        for p = n:-1:2
            T(p-1) = K(i)*sqrt(rho(p-1))*tanh(K(i)*h(p-1)/(sqrt(rho(p-1))));
            S(p-1) = tanh(K(i)*h(p-1)/(sqrt(rho(p-1))))/(K(i)*sqrt(rho(p-1)));
            Z(p-1) = (Z(p) + T(p-1))/((Z(p)*S(p-1))+1);
        end
    pa(i) = (abs(Z(1))^2)/(om(i)*mu);
    phase(i) = (180/pi)*angle(Z(1));  
end

%% PLOT

loglog(f,pa);
hold on;
set (gca,'xdir','reverse')
title('Apparent Resistivity plot');
xlabel('Frequency (Hz)');
ylabel('Apparent Resistivity (ohm.m)');
figure;
semilogx(f,phase);
hold on;
set (gca,'xdir','reverse')
title('Phase plot');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');

