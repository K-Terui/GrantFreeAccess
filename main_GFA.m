% Grant-Free Access
% Only consider the AUD (active user detection) and CE (channel estimation)
% with OMP method
% SMV considers that base station (BS) has single antenna
% MMV considers that BS has multiple antennas

% clear all; close all; clc;
clear
tic

%% method
flag_SMV = 1; %consider the single antenna BS (SMV-OMP)
flag_MMV = 0; %consider the multi antennas BS (MMV-OMP)
flag_MC  = 0; %compare the mutual coherence of each frame

%% frame
frame_GRF = 1; %gaussian random frame
frame_DFT = 0; %partial DFT frame
frame_QCS = 0; %QCSIDCO frame

%% plot
plot_FA   = 1; %plot the performance about false alarm vs. SNR
plot_MD   = 1; %plot the performance about miss detection vs. SNR
plot_NMSE = 1; %plot the performance about channel estimation error (NMSE) vs. SNR

%% setup
% parameters
N = 200; %num. UEs
M = 55; %len. spread sequence
K = 20;  %num. active UEs
J.SMV = 1;   %num. antennas of BS (SMV)
J.MMV = 16;  %num. antennas of BS (MMV)
beta  = 1;   %pathloss or shadowing components
SetUE = 1:N; %set of UEs

iter_sidco = 20; %num. of iterations for QCSIDCO algorithm

reals = 100; %num. channel realization for

% variable
SNR = -10:4:10; %SNR[dB]
Noivar = 10.^(-SNR./10) / M; %noise variance (linear value)

% K-sparse vector
S         = zeros(N, 1);
S(1:K, :) = ones (K, 1);


%% main roop
if (flag_SMV)
    % initialize
    % miss detection probability
    pmdGRFreals = ones(length(SNR), reals);
    pmdDFTreals = ones(length(SNR), reals);
    pmdQCSreals = ones(length(SNR), reals);
    % false alarm probability
    pfaGRFreals = ones(length(SNR), reals) .* K/(N-K);
    pfaDFTreals = ones(length(SNR), reals) .* K/(N-K);
    pfaQCSreals = ones(length(SNR), reals) .* K/(N-K);
    % NMSE
    nmseGRFreals = zeros(length(SNR), reals);
    nmseDFTreals = zeros(length(SNR), reals);
    nmseQCSreals = zeros(length(SNR), reals);

    Nbs = J.SMV;

    for iter = 1:reals
        for sn = 1:length(SNR)
            % active UE selection
            S_t = S(randperm(N));     %choose K active users randomly
            SetActiveUEs = find(S_t); %indeces of active UEs

            % channel generation
%             h = crand(N, Nbs) * sqrt(beta) .* repmat(S_t, 1, Nbs);
            h = sqrt(0.5) * (randn(N, Nbs) + 1j * randn(N, Nbs)) .* repmat(S_t, 1, Nbs);
            
            % noise generation
%             n = crandn(M, Nbs) * sqrt(Noivar(sn));
            n = sqrt(0.50 * Noivar(sn)) * (randn(M, Nbs) + 1j * randn(M, Nbs));

            % AUD and CE by each frame
            if (frame_GRF)
                % generate the frame
%                 A_GRF = crandn(M, N);                %gaussian random frame
                A_GRF = sqrt(0.5) * (randn(M, N) + 1j * randn(M, N));
                A_GRF = A_GRF./vecnorm(A_GRF, 2, 1); %normalize
                % received signal
%                 y = A_GRF * h + crandn(M, Nbs) * sqrt(Noivar(sn));
                y = A_GRF * h + n;

                % OMP
                % initialize
                xhat_GRF   = zeros(size(h));  %estimation value (sparse)   
                r          = zeros(size(y));  %residual
                SetEst_GRF = int16.empty;     %index set of estimated active UEs
                A          = A_GRF;           %initial frame update
                A_hat      = double.empty;    %selected support

                % main loop
                for s = 1 : K
                    % residual update
                    r = y - A * xhat_GRF;

                    % select the maximum correlated index
                    [~, p] = max(abs(A' * r));
                    
                    % add index to the index set
                    SetEst_GRF(s) = p;
                    

                    % derive the least square solution
                    A_hat(:, s) = A(:, SetEst_GRF(s));
%                     xtilde_GRF  = inv(A_hat' * A_hat) * A_hat' * y;
                    xtilde_GRF  = pinv(A_hat) * y;

                    % sparse reconstruction
                    xhat_GRF(SetEst_GRF) = xtilde_GRF;


                end
%                 [xhat_GRF, SetEst_GRF] = OrthogonalMatchingPursuit(y, A_GRF, K);

                [pmdGRFreals(sn, iter), pfaGRFreals(sn, iter), ~] = Compute_MDandFA(SetActiveUEs, SetEst_GRF, N, K);

                % MMSE
                Aest = A_GRF(:, SetEst_GRF);
                xmmseGRF = Aest' * inv(Aest * Aest' + Noivar(sn) * eye(M)) * y;
                xhat_GRF(SetEst_GRF) = xmmseGRF;

                % miss detection probability
%                 pmdGRFreals(sn, reals) = length(setdiff(SetActiveUEs, SetEst_GRF))/K;
%                 pmdGRFreals(sn, reals) = MD;
%                 % false alarm probability
%                 pfaGRFreals(sn, reals) = length(intersect(complement(SetActiveUEs, N), SetEst_GRF)) / (N-K);
                
                % NMSE
                nmseGRFreals(sn, iter) = norm(h - xhat_GRF, 'fro')^2 / norm(h, 'fro')^2;

                % SNR
                SNR_iter(sn, iter) = norm(h, 'fro')^2 / norm(n, 'fro')^2 / K;%各アクティブユーザごとのSNRに対する平均値
            end
        end
    end
    
    % calculate the expected values
    % miss detection probability
    pmdGRF = mean(pmdGRFreals, 2);
    pmdDFT = mean(pmdDFTreals, 2);
    pmdQCS = mean(pmdQCSreals, 2);
    % false alarm probability
    pfaGRF = mean(pfaGRFreals, 2);
    pfaDFT = mean(pfaDFTreals, 2);
    pfaQCS = mean(pfaQCSreals, 2);
    % NMSE
    nmseGRF = (mean(nmseGRFreals, 2));
    nmseDFT = (mean(nmseDFTreals, 2));
    nmseQCS = (mean(nmseQCSreals, 2));
    % SNR
    SNR_real = 10*log10(mean(SNR_iter, 2))
    
    %% plot
    % false alarm
    if (plot_FA)
    f1 = figure(1);
    % f1.Position(3:4) = [900 450]; % for draft
    f1.Position(3:4) = [560 420]; % for slide
    % f1.Position(3:4) = [600 350]; % for thesis
%     hold on
    

    xlabel("SNR [dB]" , "Fontsize", 15, "Fontname", "Times New Roman");
    ylabel("Probability", "Fontsize", 15, "Fontname", "Times New Roman");
    title('False Alarm and Miss Detection', 'Interpreter', 'Latex', 'Fontsize', 14);

    pfa_grf = semilogy(SNR, pfaGRF, 'k--x', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on
%     pfa_dft = semilogy(SNR, pfaDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
%     pfa_qcs = semilogy(SNR, pfaQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    pmd_grf = semilogy(SNR, pmdGRF, 'k--x', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    
    ylim([1e-3 1])
    grid on
    box on
%     legend([pfa_grf, pmd_grf], {'Gaussian (FA)', 'Gaussian (MD)'}, 'Interpreter', 'Latex', 'Location', 'NorthEast', 'Fontsize', 15);

    end

    % miss detection
    if (plot_MD)
%     f2 = figure(2);
%     % f2.Position(3:4) = [900 450]; % for draft
%     f2.Position(3:4) = [560 420]; % for slide
%     % f2.Position(3:4) = [600 350]; % for thesis
% %     hold on
%     grid on
%     box on
% %     ylim([1e-3 1])
%     pmd_grf = semilogy(SNR, pmdGRF, 'k--x', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
% %     pmd_dft = semilogy(SNR, pmdDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
% %     pmd_qcs = semilogy(SNR, pmdQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
% 
%     xlabel("SNR [dB]" , "Fontsize", 15, "Fontname", "Times New Roman");
%     ylabel("Probability", "Fontsize", 15, "Fontname", "Times New Roman");
%     title('Miss Detection', 'Interpreter', 'Latex', 'Fontsize', 14);
% 
% %     legend([pmd_grf, pmd_dft, pmd_qcs], {'Gaussian', 'DFT', 'QCSIDCO'}, 'Interpreter', 'Latex', 'Location', 'NorthEast', 'Fontsize', 15);
    
    end

    % NMSE
    if (plot_NMSE)
    f3 = figure(2);
    % f2.Position(3:4) = [900 450]; % for draft
    f3.Position(3:4) = [560 420]; % for slide
    % f2.Position(3:4) = [600 350]; % for thesis
%     hold on
%     ylim([1e-3 1])
    pnm_grf = semilogy(SNR, nmseGRF, 'k--x', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on
%     pnm_dft = plot(SNR, nmseDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
%     pnm_qcs = plot(SNR, nmseQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');

    xlabel("SNR [dB]" , "Fontsize", 15, "Fontname", "Times New Roman");
    ylabel("NMSE", "Fontsize", 15, "Fontname", "Times New Roman");
    title('NMSE', 'Interpreter', 'Latex', 'Fontsize', 14);
    ylim([1e-3 1.5])
    grid on
    box on
%     legend([pnm_grf, pnm_dft, pnm_qcs], {'Gaussian', 'DFT', 'QCSIDCO'}, 'Interpreter', 'Latex', 'Location', 'NorthEast', 'Fontsize', 15);

    end
end

toc