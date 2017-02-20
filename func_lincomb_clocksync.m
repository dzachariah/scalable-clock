function [theta_hat, Lambda_hat, s] = func_lincomb_clocksync( y, mu, H, G, Q_inv, c, X_anchor, Lambda_hat, s_prev, Lambda_x, x_bar, Perp, tol )
%
%Dave Zachariah 

%% Initialize
global M
global D
global I_D
global n

global V0

global Qperp

%Fixed quantities
[D,M]   = size(X_anchor);
I_D     = eye(D);
n       = length(y);

%Create projector: Q_inv * Pi
Qperp = Q_inv - Perp;

%Grid search
N_grid = 10;


%% TEMP: Plot
% 
% % Grid
% delta     = 5;
% x_grid    = 1-delta : 0.1 : 11+delta;
% y_grid    = 1-delta : 0.1 : 11+delta;
% % x_grid    = 6 : 0.1 : 10;
% % y_grid    = 6 : 0.1 : 10;
% V         = zeros(length(x_grid),length(y_grid));
% 
% 
% for x_idx = 1 : length(x_grid),
%     for y_idx = 1 : length(y_grid),
%         
%         %Position
%         x_set    = [x_grid(x_idx) y_grid(y_idx)]';
%         rho      = func_compute_rho( x_set, X_anchor );
%         [~,V0, V1] = func_comp_costfunctions(y,mu,G,c,x_bar,Lambda_x,x_set, rho);
%         
%         %Cost function
%         V(x_idx,y_idx) = real(log(V0)) + V1;
%         
%         
%         %DISP
%     end
% end
% 
% %Plot
% [X_grid,Y_grid] = meshgrid(x_grid, y_grid);
% contourf(Y_grid,X_grid,V,40) %NOTE:order
% colorbar


%% Compute local position estimate for epoch k
x_hat     = x_bar;
alpha_max = 10;

nn = 1;
while(1)
    
    %Compute ranges and its derivatives
    rho         = func_compute_rho( x_hat, X_anchor );
    gamma       = func_compute_gamma( x_hat, X_anchor );
    %Gamma       = func_compute_Gamma( x_hat, X_anchor );
    
    %Compute intermediate variables
    W_cost = (G'*Qperp*G) / c^2;
    w_cost = G'*Qperp*(y-mu) / c;
    
    %Compute functions and their derivatives
    [~,V0, ~]        = func_comp_costfunctions(y,mu,G,c,x_bar,Lambda_x,x_hat, rho);
    [~,gradV0,gradV1] = func_comp_gradient(W_cost,w_cost,x_bar,Lambda_x,x_hat, rho, gamma);
    
    %Search step length
    p_dir     = -(gradV0 + V0*gradV1);
    p_dir     = p_dir/norm(p_dir);
    alpha_set = linspace(0,1.2*alpha_max,N_grid);
    V_set     = zeros(N_grid,1);
    k         = 1;
    for alpha = alpha_set
        x_test   = x_hat + alpha * p_dir;
        rho      = func_compute_rho( x_test, X_anchor );
        [tmp,~,~]= func_comp_costfunctions(y,mu,G,c,x_bar,Lambda_x,x_test, rho);
        V_set(k) = tmp;
        k        = k+1;
    end
    [~,idx] = min(V_set);

    %Update estimate
    x_hat_new = x_hat + alpha_set(idx) * p_dir;
    alpha_max = norm(x_hat_new - x_hat);
    
    %TEMP:
    %hold on
    %plot( x_hat_new(1), x_hat_new(2), 'r*' )
    
    %Termination
    nn = nn + 1;
    if norm(x_hat_new - x_hat) < tol
        break
    elseif nn > 100
        break
    end
    
    %Store
    x_hat = x_hat_new;

    %TEMP: DISP
    %disp(nn)
    %disp(x_hat_new)
    
        
    
end


%% Compute remaining estimates
warning off

%Estimate from epoch k
x_check = x_hat_new;

%Clock parameters
rho     = func_compute_rho( x_check, X_anchor );
c_check = (H'*Q_inv*H) \ ((H'*Q_inv) * (y - mu - G*rho / c));

%Theta
theta_check = [c_check; x_check];

%Noise variance
sigma2_check = max(V0,1e-8); %TODO: check constant

%Fisher info
Gamma = func_compute_gamma( x_check, X_anchor )'; %transpose
L     = [H,  G*Gamma/c];
J_hat = (L'*Q_inv*L)/ sigma2_check;


%% Linearly combine estimates
s          = (s_prev + J_hat * theta_check);
Lambda_hat = Lambda_hat + J_hat;
theta_hat  = Lambda_hat\s;
warning on

%TEMP
%disp([theta_check theta_hat])

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute rho and its derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [rho] = func_compute_rho( x_hat, X_anchor )

global M

rho = zeros(M,1);
for m = 1:M
   rho(m) = norm( x_hat - X_anchor(:,m) );
end

end



function [gamma] = func_compute_gamma( x_hat, X_anchor )

global D
global M
gamma = zeros(D,M);
for m = 1:M
        gamma(:,m) = ( x_hat - X_anchor(:,m) )  / (norm( x_hat - X_anchor(:,m) ));
end

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute functions and derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [V, V0, V1] = func_comp_costfunctions( y, mu, G, c, x_bar, Lambda_x, x_hat, rho )

global n
global Qperp

V0 = (y - mu - G*rho/c )' * Qperp * ( y - mu - G*rho/c ) / n; 
V1 = (x_hat - x_bar)'* Lambda_x *(x_hat - x_bar) / n;
V  = real(log(V0)) + V1;

end


function [gradV, gradV0, gradV1] = func_comp_gradient(W, w, x_bar, Lambda_x, x_hat, rho, gamma)

global D
global M
global n
global V0

gradV1 = 2 * Lambda_x * (x_hat - x_bar) / n;
gradV0 = zeros(D,1);

for i = 1:M
   for j = 1:M
       gradV0 = gradV0 + W(i,j) * ( gamma(:,i)*rho(j)  +  rho(i)*gamma(:,j) );
   end
   
   gradV0 = gradV0 - 2*w(i)*gamma(:,i);
   
end

gradV = gradV0/V0 + gradV1;

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unused functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [Gamma] = func_compute_Gamma( x_hat, X_anchor )
% 
% global D
% global M
% global I_D
% Gamma = zeros(D,D,M);
% for m = 1:M
%     Gamma(:,:,m) = I_D/norm(x_hat - X_anchor(:,m)) ... 
%                    - (x_hat - X_anchor(:,m))*(x_hat - X_anchor(:,m))' / (norm(x_hat - X_anchor(:,m))^3);
% end
% 
% end

% function [hessV0, hessV1] = func_comp_hessian(W, w, Lambda_x)
% 
% global D
% global M
% global n
% global rho
% global gamma
% global Gamma
% 
% hessV1 = 2 * Lambda_x / n;
% hessV0 = zeros(D,D);
% 
% for i = 1:M
%    for j = 1:M
%        
%        Gam_collect = ( Gamma(:,:,i)*rho(j) + gamma(:,j)*gamma(:,i)' + ...
%                                              gamma(:,i)*gamma(:,j)' + rho(i)*Gamma(:,:,j)  );
%        
%        hessV0 = hessV0 + W(i,j) * Gam_collect;
%                                                       
%    end
%       
%    hessV0 = hessV0 - 2*w(i)*Gamma(:,:,i);
%    
% end
% 
% 
% end


% 
% function [Jac] = func_compute_rangejacobian( x_u_hat, X_anchor )
% 
% global D
% global M
% Jac = zeros(M,D);
% for m = 1:M
%         Jac(m,:) = ( x_u_hat - X_anchor(:,m) )'  / (norm( x_u_hat - X_anchor(:,m) ));
% end
% 
% end
% 
% 
% function [Gamma] = func_compute_Gamma( x_u_hat, X_anchor )
% 
% global D
% global M
% global I_D
% Gamma = zeros(D,D,M);
% for m = 1:M
%     Gamma(:,:,m) = I_D/norm(x_u_hat - X_anchor(:,m)) ... 
%                    - (x_u_hat - X_anchor(:,m))*(x_u_hat - X_anchor(:,m))' / (norm(x_u_hat - X_anchor(:,m))^3);
% end
% 
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%
% % Compute Gradients
% %%%%%%%%%%%%%%%%%%%%%%%
