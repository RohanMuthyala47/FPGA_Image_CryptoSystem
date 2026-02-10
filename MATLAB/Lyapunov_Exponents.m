%% fractional_chen4d_LE.m
% Fractional-Order 4D Chen Hyperchaotic System with Lyapunov Exponent Calculation
% Single-file implementation: main script + helper functions.
% Save as fractional_chen4d_LE.m and run.

clc; clear; close all;

%% ======================== PARAMETERS ====================================

% System parameters (from paper)
alpha   = 35;
gamma_c = 28;    % renamed from gamma -> gamma_c to avoid shadowing MATLAB's gamma()
epsilon = 12;
beta    = 3;
lambda  = 0.5;

% Initial conditions (from paper)
x0 = 0;
y0 = 0;
z0 = 8;
w0 = 6;

% Fractional orders (from paper)
q = [0.95, 0.95, 0.95, 0.95];  % q1, q2, q3, q4

% Time parameters
h = 0.01;           % Step size
T_end = 200;        % Total time
N = round(T_end/h); % Number of iterations

fprintf('========================================\n');
fprintf('Fractional-Order 4D Chen System\n');
fprintf('========================================\n');
fprintf('Parameters: α=%.1f, γ=%.1f, ε=%.1f, β=%.1f, λ=%.1f\n', ...
    alpha, gamma_c, epsilon, beta, lambda);
fprintf('Initial: [x0, y0, z0, w0] = [%.1f, %.1f, %.1f, %.1f]\n', ...
    x0, y0, z0, w0);
fprintf('Fractional orders: q = [%.2f, %.2f, %.2f, %.2f]\n', q);
fprintf('Time: h=%.3f, N=%d\n', h, N);
fprintf('========================================\n\n');

%% =============== FRACTIONAL-ORDER SYSTEM INTEGRATION ===================

fprintf('Integrating fractional-order system...\n');
tic;

[t, X] = fractional_chen4d_ABM(x0, y0, z0, w0, alpha, gamma_c, epsilon, ...
                                beta, lambda, q, h, N);

integration_time = toc;
fprintf('Integration completed in %.2f seconds\n\n', integration_time);

x = X(:,1); y = X(:,2); z = X(:,3); w = X(:,4);

%% ================= LYAPUNOV EXPONENTS CALCULATION ======================

fprintf('Calculating Lyapunov Exponents...\n');
tic;

[LE, LE_track, t_track] = calculate_lyapunov_exponents_fractional(...
    x0, y0, z0, w0, alpha, gamma_c, epsilon, beta, lambda, q, h, N);

le_time = toc;
fprintf('Lyapunov calculation completed in %.2f seconds\n\n', le_time);

fprintf('========================================\n');
fprintf('LYAPUNOV EXPONENTS:\n');
fprintf('========================================\n');
fprintf('LE1 = %+.6f\n', LE(1));
fprintf('LE2 = %+.6f\n', LE(2));
fprintf('LE3 = %+.6f\n', LE(3));
fprintf('LE4 = %+.6f\n', LE(4));
fprintf('========================================\n');
fprintf('Sum of LEs = %.6f\n', sum(LE));

% Check for hyperchaotic behavior
positive_LEs = sum(LE > 0);
fprintf('Number of positive LEs: %d\n', positive_LEs);
if positive_LEs >= 2
    fprintf('✓ System exhibits HYPERCHAOTIC behavior\n');
elseif positive_LEs == 1
    fprintf('System exhibits chaotic behavior\n');
else
    fprintf('System is NOT chaotic\n');
end
fprintf('========================================\n\n');

%% ====================== CORRECTED LE PLOT ==============================
% LE_track has size (4 × K)
% t_track has size (1 × K)

figure('Name','Lyapunov Exponent Evolution','NumberTitle','off');
hold on;
plot(t_track, LE_track(1,:), 'LineWidth', 1.5);
plot(t_track, LE_track(2,:), 'LineWidth', 1.5);
plot(t_track, LE_track(3,:), 'LineWidth', 1.5);
plot(t_track, LE_track(4,:), 'LineWidth', 1.5);

xlabel('Time','FontSize',14);
ylabel('Lyapunov Exponent','FontSize',14);
title('Lyapunov Exponent Evolution','FontSize',16);
legend('LE1','LE2','LE3','LE4','Location','Best');
grid on;
hold off;

% Annotate final values at final t_track
if ~isempty(t_track)
    t_annot = t_track(end);
    y1 = LE_track(1,end);
    y2 = LE_track(2,end);
    y3 = LE_track(3,end);
    y4 = LE_track(4,end);

    text(t_annot, y1, sprintf(' LE1 = %.4f', LE(1)), ...
         'FontSize',12,'Color','b','BackgroundColor','w');

    text(t_annot, y2, sprintf(' LE2 = %.4f', LE(2)), ...
         'FontSize',12,'Color','r','BackgroundColor','w');

    text(t_annot, y3, sprintf(' LE3 = %.4f', LE(3)), ...
         'FontSize',12,'Color',[1 0.5 0],'BackgroundColor','w');

    text(t_annot, y4, sprintf(' LE4 = %.4f', LE(4)), ...
         'FontSize',12,'Color',[0.6 0 0.9],'BackgroundColor','w');
end

%% ========================= VISUALIZATION (phase plots) =================
% Example 3D phase plot (x,y,z)
figure('Name','Phase plot x-y-z','NumberTitle','off');
plot3(x, y, z, 'LineWidth', 0.5);
xlabel('x'); ylabel('y'); zlabel('z');
grid on; title('Phase plot: x-y-z');

%% ======================= HELPER FUNCTIONS ==============================

%% Function: Fractional-Order 4D Chen System Integration
function [t, X] = fractional_chen4d_ABM(x0, y0, z0, w0, alpha, gamma_c, ...
                                        epsilon, beta, lambda, q, h, N)
    % Predictor-Corrector ABM for Fractional ODEs (per-dim binomial weights)
    X = zeros(N+1,4);
    X(1,:) = [x0, y0, z0, w0];
    t = (0:N) * h;

    % precompute binomial weights for each dimension
    c = zeros(4, N+1);
    for i = 1:4
        c(i,:) = compute_binomial_coeffs(q(i), N);
    end

    % System function
    F = @(s) [ alpha*(s(2)-s(1)) + s(4);
               gamma_c*s(1) - s(1)*s(3) + epsilon*s(2);
               s(1)*s(2) - beta*s(3);
               s(2)*s(3) + lambda*s(4) ];

    qv = q(:); % 4x1

    fprintf('Progress: ');
    last_percent = -1;
    for n = 1:N
        % predictor using history
        predictor = zeros(4,1);
        for j = 1:n
            fj = F(X(j,:)') ; % 4x1
            predictor = predictor + ( [c(1,n-j+2); c(2,n-j+2); c(3,n-j+2); c(4,n-j+2)] .* fj );
        end

        state = X(n,:)';
        X_pred = state + (h.^qv) .* predictor ./ gamma(qv + 1);

        % evaluate f at predictor
        f_pred = F(X_pred);

        % corrector sum
        corrector = zeros(4,1);
        for j = 1:n
            fj = F(X(j,:)');
            corrector = corrector + ( [c(1,n-j+1); c(2,n-j+1); c(3,n-j+1); c(4,n-j+1)] .* fj );
        end

        % final update (element-wise)
        X_next = state + (h.^qv) ./ gamma(qv + 2) .* f_pred + ...
                 (h.^qv) ./ gamma(qv + 2) .* corrector;

        X(n+1,:) = X_next';

        % progress
        percent = floor(100 * n / N);
        if percent ~= last_percent && mod(percent,10) == 0
            fprintf('%d%% ', percent);
            last_percent = percent;
        end
    end
    fprintf('100%%\n');
end

%% Function: Compute Binomial Coefficients
function c = compute_binomial_coeffs(q, N)
    % Uses general binomial coefficients: w_k = (-1)^k * (q choose k)
    % where (q choose k) = gamma(q+1)/(gamma(k+1)*gamma(q-k+1))
    c = zeros(N+1,1);
    for k = 0:N
        c(k+1) = (-1)^k * gamma(q + 1) / ( gamma(k + 1) * gamma(q - k + 1) );
    end
end

%% Function: Calculate Lyapunov Exponents
function [LE, LE_track, t_track] = calculate_lyapunov_exponents_fractional(...
    x0, y0, z0, w0, alpha, gamma_c, epsilon, beta, lambda, q, h, N)

    % initialize state and tangent basis
    X = [x0; y0; z0; w0];
    V = eye(4);
    LE_accum = zeros(4,1);

    % tracking
    track_interval = 100;                  % adjust as desired
    n_track = floor(N / track_interval);
    if n_track < 1
        LE_track = zeros(4,0);
        t_track = zeros(1,0);
    else
        LE_track = zeros(4, n_track);
        t_track = zeros(1, n_track);
    end
    track_idx = 1;

    % binomial weights and X_history for fractional integrator
    X_history = zeros(N+1, 4);
    X_history(1,:) = X';
    c = zeros(4, N+1);
    for i = 1:4
        c(i,:) = compute_binomial_coeffs(q(i), N);
    end

    % system and Jacobian
    F = @(s) [ alpha*(s(2)-s(1)) + s(4);
               gamma_c*s(1) - s(1)*s(3) + epsilon*s(2);
               s(1)*s(2) - beta*s(3);
               s(2)*s(3) + lambda*s(4) ];

    J_func = @(s) [ -alpha,            alpha,         0,      1;
                    gamma_c - s(3),    epsilon,      -s(1),  0;
                    s(2),              s(1),         -beta,  0;
                    0,                 s(3),         s(2),   lambda ];

    qv = q(:);

    fprintf('Progress: ');
    last_percent = -1;
    for n = 1:N
        % -------------- fractional state propagation (ABM) --------------
        predictor = zeros(4,1);
        for j = 1:n
            fj = F( X_history(j,:)' );
            predictor = predictor + ( [c(1,n-j+2); c(2,n-j+2); c(3,n-j+2); c(4,n-j+2)] .* fj );
        end
        X_pred = X + (h.^qv) .* predictor ./ gamma(qv + 1);
        f_pred = F(X_pred);
        corrector = zeros(4,1);
        for j = 1:n
            fj = F( X_history(j,:)' );
            corrector = corrector + ( [c(1,n-j+1); c(2,n-j+1); c(3,n-j+1); c(4,n-j+1)] .* fj );
        end
        X = X + (h.^qv) ./ gamma(qv + 2) .* f_pred + ...
                (h.^qv) ./ gamma(qv + 2) .* corrector;
        X_history(n+1,:) = X';

        % -------------- propagate tangent vectors (linearized) --------------
        J = J_func(X);
        % matrix exponential propagation for the linearized flow over step h
        Phi = expm(h * J);   % 4x4
        V = Phi * V;

        % orthonormalize & accumulate every 10 steps
        if mod(n, 10) == 0
            [Q, R] = qr(V);
            V = Q;
            R_diag = abs(diag(R));
            R_diag(R_diag < eps) = eps;
            LE_accum = LE_accum + log(R_diag);
        end

        % track LEs at intervals
        if n_track >= 1 && mod(n, track_interval) == 0
            current_time = n*h;
            LE_track(:, track_idx) = LE_accum / current_time;
            t_track(track_idx) = current_time;
            track_idx = track_idx + 1;
        end

        % progress
        percent = floor(100 * n / N);
        if percent ~= last_percent && mod(percent,10) == 0
            fprintf('%d%% ', percent);
            last_percent = percent;
        end
    end
    fprintf('100%%\n');

    total_T = (n) * h;
    LE = LE_accum / total_T;
end
