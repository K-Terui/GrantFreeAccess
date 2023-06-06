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
frame_DFT = 1; %partial DFT frame
frame_QCS = 0; %QCSIDCO frame

%% plot
plot_MDFA = 1; %plot the performance about miss detection and false alarm vs. SNR
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
    % SNR
    SNRreals = zeros(length(SNR), reals);

    Nbs = J.SMV;

    for iter = 1:reals
        for sn = 1:length(SNR)
            % active UE selection
            S_t = S(randperm(N));     %choose K active users randomly
            SetActiveUEs = find(S_t); %indeces of active UEs

            % channel generation
            h = sqrt(0.5) * (randn(N, Nbs) + 1j * randn(N, Nbs)) .* repmat(S_t, 1, Nbs);
            
            % noise generation
            n = sqrt(0.50 * Noivar(sn)) * (randn(M, Nbs) + 1j * randn(M, Nbs));
            % SNR
            SNRreals(sn, iter) = norm(h, 'fro')^2 / norm(n, 'fro')^2 / K; %各アクティブユーザごとのSNRに対する平均値

            % AUD and CE by each frame
            % gaussian random frame
            if (frame_GRF)
                % generate the frame
                A_GRF = sqrt(0.5) * (randn(M, N) + 1j * randn(M, N));
                A_GRF = A_GRF./vecnorm(A_GRF, 2, 1); %normalization
                
                % received signal
                y = A_GRF * h + n;

                % OMP
                [xhat_GRF, SetEst_GRF] = OMP_nzknown(y, A_GRF, K);
                
                % MD, FA, AER
                [pmdGRFreals(sn, iter), pfaGRFreals(sn, iter), ~] = Compute_MDandFA(SetActiveUEs, SetEst_GRF, N, K);

                % MMSE
                nmseGRFreals(sn, iter) = ChannelEstimation_MMSE(A_GRF, y, h, xhat_GRF, SetEst_GRF, Noivar(sn));

            end
            
            % partial DFT frame
            if (frame_DFT)
                % generate the frame
                dftmat = dftmtx(N) / sqrt(N);
                A_DFT = dftmat(randperm(N, M), :);
                A_DFT = A_DFT./vecnorm(A_DFT, 2, 1); %normalization

                % received signal
                y = A_DFT * h + n;

                % OMP
                [xhat_DFT, SetEst_DFT] = OMP_nzknown(y, A_DFT, K);
                
                % MD, FA, AER
                [pmdDFTreals(sn, iter), pfaDFTreals(sn, iter), ~] = Compute_MDandFA(SetActiveUEs, SetEst_DFT, N, K);

                % MMSE
                nmseDFTreals(sn, iter) = ChannelEstimation_MMSE(A_DFT, y, h, xhat_DFT, SetEst_DFT, Noivar(sn));

            end

            % QCSIDCO frame
            if (frame_QCS)
                % frame is pre-generated

                % received signal
                y = A_QCS * h + n;

                % OMP
                [xhat_QCS, SetEst_QCS] = OMP_nzknown(y, A_QCS, K);
                
                % MD, FA, AER
                [pmdQCSreals(sn, iter), pfaQCSreals(sn, iter), ~] = Compute_MDandFA(SetActiveUEs, SetEst_QCS, N, K);

                % MMSE
                nmseQCSreals(sn, iter) = ChannelEstimation_MMSE(A_QCS, y, h, xhat_QCS, SetEst_QCS, Noivar(sn));
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
    SNRtrue = 10*log10(mean(SNRreals, 2));

    
    %% plot
    % false alarm and miss detection
    if (plot_MDFA)
    f1 = figure(1);
    % f1.Position(3:4) = [900 450]; % for draft
    f1.Position(3:4) = [560 420]; % for slide
    % f1.Position(3:4) = [600 350]; % for thesis
%     hold on
    

    xlabel("SNR [dB]" , "Fontsize", 15, "Fontname", "Times New Roman");
    ylabel("Probability", "Fontsize", 15, "Fontname", "Times New Roman");
    title('False Alarm and Miss Detection', 'Interpreter', 'Latex', 'Fontsize', 14);

    
    pmd_grf = semilogy(SNR, pmdGRF, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on
    pmd_dft = semilogy(SNR, pmdDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
%     pmd_qcs = semilogy(SNR, pmdQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');

    pfa_grf = semilogy(SNR, pfaGRF, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    pfa_dft = semilogy(SNR, pfaDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
%     pfa_qcs = semilogy(SNR, pfaQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    

    ylim([1e-3 1e-0])
    grid on
    box on
    legend([pmd_grf, pmd_dft, pfa_grf, pfa_dft], {'Gaussian (MD)', 'Partial DFT (MD)', 'Gaussian (FA)','Partial DFT (FA)'}, 'Interpreter', 'Latex', 'Location', 'southwest', 'Fontsize', 15);

    end


    % NMSE
    if (plot_NMSE)
    f2 = figure(2);
    % f2.Position(3:4) = [900 450]; % for draft
    f2.Position(3:4) = [560 420]; % for slide
    % f2.Position(3:4) = [600 350]; % for thesis

    pnm_grf = semilogy(SNR, nmseGRF, 'k--x', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on
    pnm_dft = plot(SNR, nmseDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
%     pnm_qcs = plot(SNR, nmseQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');

    xlabel("SNR [dB]" , "Fontsize", 15, "Fontname", "Times New Roman");
    ylabel("NMSE", "Fontsize", 15, "Fontname", "Times New Roman");
    title('NMSE', 'Interpreter', 'Latex', 'Fontsize', 14);
    ylim([1e-3 1e0])
    grid on
    box on
    legend([pnm_grf, pnm_dft], {'Gaussian', 'Partial DFT'}, 'Interpreter', 'Latex', 'Location', 'southwest', 'Fontsize', 15);

    end
end

toc