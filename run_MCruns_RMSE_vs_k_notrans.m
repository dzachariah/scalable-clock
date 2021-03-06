clear
close all

%% Seed
%RandStream.setGlobalStream(RandStream('mt19937ar','seed',1)); %divergence!
%RandStream.setGlobalStream(RandStream('mt19937ar','seed',3)); %normal!

RandStream.setGlobalStream(RandStream('mt19937ar','seed','shuffle'));


%% Initalize

%Monte Carlo runs
MC_runs = 1000;
MC_runs_crb = 10;

%Number of epochs
K_epochs = 1e3;

%Epochs with transceiving nodes
epoch_auxnodes = zeros(K_epochs,1);

%True node positions
X = [  1,   1;
      11,  11;
       1,  11;
      11,   1;
       9,   8]'
  
%Initial position estimate
sigma2_x = (2e-1)^2;
Lambda_x = eye(2) / sigma2_x;          
x_init   = X(:,5);

% True values
Tm         = 50e-9;
Tu         = 50e-9;
phi_u      = Tu/10;
clock_true = [phi_u Tu Tm ]';

theta_true = [clock_true; zeros(2,1)];

%Noise level (base)
sigma2_0 = (2e-9)^2;

% Epoch update rate
N = 101;
M = 100;

% Delay at nodes while replying in seconds
delta = 10^(-6);

%Propagation velocity
c = 3*10^8;

%Estimator tolerance
tol = 1e-7;


%% Store results
steps_k_est    = 10;
sq_error_theta = zeros(3+2,steps_k_est, MC_runs);

%% Construct measurement
%Anchors
X_anchor = X(:,1:end-1);

%Distances
rho = pdist(X','euclidean');
rho = squareform(rho);

%Create variables
mu0   = [0, 0, 0, rho(1,2)/c+delta, rho(2,3)/c+delta, rho(3,4)/c+delta]';


%Full covariance matrix
sigma2_set = sigma2_0 * ones(K_epochs,1);
alpha2     = (1/10)^2;
Q0         = [(1+alpha2),   0,      1, 0, 0, 0;
                       0, 2*alpha2, 0, 0, 0, 0;
                       1,   0,      2, 1, 0, 0;
                       0,   0,      1, 2, 1, 0;
                       0,   0,      0, 1, 2, 1;
                       0,   0,      0, 0, 1, 2]; 

H0     = [1, 0, 0;
          0, N, 0;
          0, 0, M;
          0, 0, 0;
          0, 0, 0;
          0, 0, 0];
G0     = [-1, 0, 0, 0;
           0, 0, 0, 0;
           0, 0, 0, 0;
          -1, 1, 0, 0;
           0,-1, 1, 0;
           0, 0,-1, 1];               
                   
       
%% Compute RMSE
tic 
%Initialize
k_set_est  = round(10.^linspace(0,log10(K_epochs),steps_k_est));
count      = 1;

for m = 1:MC_runs
    
    %Random position
    x_true          = X(:,5) + sqrt(sigma2_x)*randn(2,1);
    theta_true(4:5) = x_true;
    
    %Initialize estimator
    theta_hat = zeros(5,1); theta_hat(4:5) = x_init;
    Lambda    = zeros(5,5); Lambda(4:5,4:5) = Lambda_x;
    s         = Lambda*theta_hat;
        
    %Varying rho
    rho   = pdist([X_anchor x_true]','euclidean');
    rho   = squareform(rho);
    rho_x = [rho(5,1), rho(5,2), rho(5,3), rho(5,4)]';
    
    
    for k = 1:K_epochs
        
        %Selection matrix
        if epoch_auxnodes(k) == 1
            S         = eye(6);
            n_samples = 6;
        else
            S         = [eye(3), zeros(3,3)];
            n_samples = 3;
        end
        
        %Model variables
        mu      = S*mu0;
        H0(1,:) = [1, (k-1)*N, -(k-1)*M];
        H       = S*H0;
        G       = S*G0;
        Q       = S*Q0*S';
        Q_sqrt  = chol(Q, 'lower');
        Q_inv   = Q \ eye(n_samples);
        
        %Create numerically stable projector
        warning off
        [V,diagD] = eig(H * ((H'*Q_inv*H) \ H'));
        S         = (Q_inv*V);
        Perp      = (S*diagD*S');
        warning on
        
        %Generate measurement
        y = mu + (H/c)*(clock_true*c) + (G*rho_x / c) + sqrt(sigma2_set(k)) * Q_sqrt * randn(n_samples,1);
        
        %Estimate
        [theta_hat, Lambda, s] = func_lincomb_clocksync(  y, mu, H, G, Q_inv, c, X_anchor, Lambda, s,  Lambda_x, x_init, Perp, tol );

        
        %Record error
        if ismember(k,k_set_est)
            sq_error_theta(:,count,m) = (theta_true - theta_hat).^2;
            count = count + 1;
            %disp(k)
        end
    end
    
    count = 1;
    disp('---------------')
    disp(m)
    disp('---------------')
end
toc

%Compute MSE
MSE_theta = mean( sq_error_theta, 3 );
                   
%% Plot Cram�r-Rao bound w.r.t. position (RMSE) in [ns]
[D,~]       = size(X);
[N_theta,~] = size(theta_true);

warning('off','all')
% Grid
steps_k = 1000;
k_set   = round(10.^linspace(0,log10(K_epochs),steps_k));

CRB_clk   = zeros(3,length(k_set));
idx_clk   = [1:3];
idx_set   = [4:5];
idx_k     = 1;
Lambda    = zeros(5,5);


for k = k_set
    
    %Reset
    Lambda       = zeros(5,5);
    Lambda_prior = [zeros(3,5); zeros(2,3), Lambda_x];
    
    for m = 1:MC_runs_crb
        %Generate true position around prior
        x_u = X(:,5) + sqrt(sigma2_x) * randn(2,1);
        
        %Information matrix
        Lambda  = Lambda + func_compute_fisher( x_u, X(:,1:4), sigma2_set, Q0, c, N, M, k, epoch_auxnodes, D, N_theta );
    end
    
    %Add prior information
    Lambda = Lambda/MC_runs_crb; %expected
    Lambda = Lambda + Lambda_prior;
    
    %(Marginal) Fisher information of clock parameters
    Lambda_clock = Lambda(idx_clk,idx_clk) - Lambda(idx_set,idx_clk)'*(Lambda(idx_set,idx_set)\Lambda(idx_set,idx_clk));
    
    %Cram�r-Rao bound
    CRB_clk(:,idx_k) = diag(Lambda_clock\eye(3)); %MSE in [s^2]
    idx_k            = idx_k + 1;
        
end     

%% Save
save MC_result_RMSE_vs_k_notrans_new


%% Plot
figure
loglog( k_set, sqrt(CRB_clk(1,:)), 'k-','LineWidth', 1.6), grid on, hold on
loglog( k_set, sqrt(CRB_clk(2,:)), 'b--','LineWidth', 1.6),
loglog( k_set, sqrt(CRB_clk(3,:)), 'r-.','LineWidth', 1.6),
ylabel('RMSE [s]','Interpreter','LaTex'), xlabel('\# epochs','Interpreter','LaTex')
legend_latex = legend('$\phi_u$','$T_u$','$T_m$');
set(legend_latex,'Interpreter','LaTex');

loglog( k_set_est, sqrt(MSE_theta(1,:)), 'k+','LineWidth', 1.6),
loglog( k_set_est, sqrt(MSE_theta(2,:)), 'bo','LineWidth', 1.6),
loglog( k_set_est, sqrt(MSE_theta(3,:)), 'rx','LineWidth', 1.6),

figure, grid on, hold on;
plot(X(1,1:4), X(2,1:4), 'r*', 'MarkerSize', 10, 'LineWidth', 1.8)
plot(theta_hat(4), theta_hat(5), 'k*', 'MarkerSize', 10, 'LineWidth', 1.8)
axis([0 12  0 12]);

% figure
% [X_grid,Y_grid] = meshgrid(x_grid, y_grid);
% contourf(Y_grid,X_grid,CRB_phi,50)
% %contourf(Y_grid,X_grid,V,linspace(min(V(:)), 3, 20))
% hold on;
% plot(X(1,1:4), X(2,1:4), 'r*', 'MarkerSize', 10, 'LineWidth', 1.8)
% colorbar
% axis([min(x_grid) max(x_grid)  min(y_grid) max(y_grid)]);
% %axis equal
% drawnow;
% xlabel('$x_1$ [m]','Interpreter','latex'), ylabel('$x_2$ [m]','Interpreter','latex')
% warning('on','all')