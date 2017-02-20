clear
close all


%% Seed
%RandStream.setGlobalStream(RandStream('mt19937ar','seed',1)); %divergence!
%RandStream.setGlobalStream(RandStream('mt19937ar','seed',3)); %normal!

RandStream.setGlobalStream(RandStream('mt19937ar','seed','shuffle'));



%% Initalize

%Monte Carlo runs
MC_runs = 1000;
MC_runs_crb = 100;


%Number of epochs
K_epochs = 5e2;

%Epochs with transceiving nodes
epoch_auxnodes = zeros(K_epochs,1);

%True node positions
X = [  1,   1;
      11,  11;
       1,  11;
      11,   1;
       9,   8]'
  
%Position precision
%Lambda_x = zeros(2,2);          
x_init   = X(:,5);

% True values
Tm         = 50e-9;
Tu         = 50e-9;
phi_u      = Tu/10;%1e-9;
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
steps_k_est    = 8;
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
sigma2_set = sigma2_0 * ones(K_epochs,1);%(5e-9)^2;
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

%Initialize
%sigma2_x_set = (10.^linspace(-2,0,steps_k_est)).^2;
sigma2_x_set = linspace(1e-2,1e0,steps_k_est).^2;
count        = 1;



for m = 1:MC_runs
    
    
    %Randomize
    rand_x = randn(2,1);

    for sigma2_x = sigma2_x_set
        
        %Random position
        x_true          = X(:,5) + sqrt(sigma2_x)*rand_x;
        theta_true(4:5) = x_true;
        
        %Varying rho
        rho   = pdist([X_anchor x_true]','euclidean');
        rho   = squareform(rho);
        rho_x = [rho(5,1), rho(5,2), rho(5,3), rho(5,4)]';
        
        %Initialize estimator
        theta_hat = zeros(5,1); theta_hat(4:5) = X(:,5);
        Lambda    = zeros(5,5); Lambda(4:5,4:5) = eye(2)/sigma2_x;
        s         = Lambda*theta_hat;
        
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
            y = mu + H*clock_true + (G*rho_x / c) + sqrt(sigma2_set(k)) * Q_sqrt * randn(n_samples,1);
            
            %Estimate
           %[theta_hat, Lambda] = func_map_clocksync( y, mu, H, G, Q_inv, c, X_anchor, tol, Lambda, theta_hat );
            [theta_hat, Lambda, s] = func_lincomb_clocksync(  y, mu, H, G, Q_inv, c, X_anchor, Lambda, s,  Lambda(4:5,4:5), x_init, Perp, tol );

            
        end
        
        %Record error
        sq_error_theta(:,count,m) = (theta_true - theta_hat).^2;
        
        disp(sqrt(sq_error_theta(1,count,m)))
        
        count = count + 1;
        
    end
    count = 1;
    disp('---------------')
    disp(m)
    
    disp('---------------')
end

%Compute MSE
MSE_theta = mean( sq_error_theta, 3 );




%% Plot Cramér-Rao bound w.r.t. position (RMSE) in [ns]
[D,~]       = size(X);
[N_theta,~] = size(theta_true);

warning('off','all')
% Grid
steps_s2_x   = 1e2;
%sigma2_x_crb = (10.^linspace(-2,0,steps_s2_x)).^2;
sigma2_x_crb = linspace(1e-2,1e0,steps_s2_x).^2;

CRB_clk   = zeros(3,length(sigma2_x_crb));
idx_clk   = [1:3];
idx_set   = [4:5];
idx_s     = 1;



for sigma2_x = sigma2_x_crb
    
    %Reset
    Lambda       = zeros(5,5);
    Lambda_prior = [zeros(3,5); zeros(2,3), eye(2)/sigma2_x];
    
    for m = 1:MC_runs_crb
        %Generate true position around prior
        x_u = X(:,5) + sqrt(sigma2_x) * randn(2,1);
        
        %Information matrix
        Lambda  = Lambda + func_compute_fisher( x_u, X(:,1:4), sigma2_set, Q0, c, N, M, K_epochs, epoch_auxnodes, D, N_theta );
    end
    
    %Add prior information
    Lambda = Lambda/MC_runs_crb; %expected
    Lambda = Lambda + Lambda_prior;
    
    %(Marginal) Fisher information of clock parameters
    Lambda_clock = Lambda(idx_clk,idx_clk) - Lambda(idx_set,idx_clk)'*(Lambda(idx_set,idx_set)\Lambda(idx_set,idx_clk));
    
    %Cramér-Rao bound
    CRB_clk(:,idx_s) = diag(Lambda_clock\eye(3)); %MSE in [s^2]
    idx_s            = idx_s + 1;
    
end


%% Save
save MC_result_RMSE_vs_sigma2_x_notrans_new

%% Plot
% figure
% loglog( sqrt(sigma2_x_crb), sqrt(CRB_clk(1,:)), 'k-','LineWidth', 1.6), grid on, hold on
% loglog( sqrt(sigma2_x_crb), sqrt(CRB_clk(2,:)), 'b--','LineWidth', 1.6),
% loglog( sqrt(sigma2_x_crb), sqrt(CRB_clk(3,:)), 'r-.','LineWidth', 1.6),
% ylabel('RMSE [s]','Interpreter','LaTex'), xlabel('$\sigma_x$ [m]','Interpreter','LaTex')
% legend_latex = legend('$\phi$','$T_u$','$T_m$');
% set(legend_latex,'Interpreter','LaTex');
% 
% loglog( sqrt(sigma2_x_set), sqrt(MSE_theta(1,:)), 'k+','LineWidth', 1.6),
% loglog( sqrt(sigma2_x_set), sqrt(MSE_theta(2,:)), 'bo','LineWidth', 1.6),
% loglog( sqrt(sigma2_x_set), sqrt(MSE_theta(3,:)), 'rx','LineWidth', 1.6),


figure
semilogy( sqrt(sigma2_x_crb), sqrt(CRB_clk(1,:)), 'k-','LineWidth', 1.6), grid on, hold on
semilogy( sqrt(sigma2_x_crb), sqrt(CRB_clk(2,:)), 'b--','LineWidth', 1.6),
semilogy( sqrt(sigma2_x_crb), sqrt(CRB_clk(3,:)), 'r-.','LineWidth', 1.6),
ylabel('RMSE [s]','Interpreter','LaTex'), xlabel('$\sigma_x$ [m]','Interpreter','LaTex')
legend_latex = legend('$\phi_u$','$T_u$','$T_m$');
set(legend_latex,'Interpreter','LaTex');

semilogy( sqrt(sigma2_x_set), sqrt(MSE_theta(1,:)), 'k+','LineWidth', 1.6),
semilogy( sqrt(sigma2_x_set), sqrt(MSE_theta(2,:)), 'bo','LineWidth', 1.6),
semilogy( sqrt(sigma2_x_set), sqrt(MSE_theta(3,:)), 'rx','LineWidth', 1.6),
