% Grant-Free Access
% Only consider the AUD (active user detection) and CE (channel estimation)
% with OMP method
% SMV considers that base station (BS) has single antenna
% MMV considers that BS has multiple antennas
% 
% Demodulation method
% MMV-OMP considers the correlation between antenna space
% P-OMP doesn't consider the antenna space correlation
% MMV-OMP : Multi measurement vector orthogonal matching pursuit 
% P-OMP : Parallel OMP
%
% Reference
% Koji Ishibashi, "Grant Free Access Tutorial" (AWCC, UEC, Tokyo)
% https://drive.google.com/file/d/1A6Lmjf9jeeezoLP-an7Ho8HCwvXO1w_b/view
% https://drive.google.com/file/d/1R3cEsMbI0DHSYIXKjrn-refiPEfhFbkA/view
%
% Edited by Kanta Terui, 15/Jun./2023 (AWCC, UEC, Tokyo)
%

% clear all; close all; clc;
clear
tic

%% method
flag_SMV = 1; %single measurement vector
flag_MMV = 1; %multi measurement vector

%% frame
frame_GRF = 1; %gaussian random frame
frame_DFT = 1; %partial DFT frame
frame_QCS = 1; %QCSIDCO frame

%% plot
plot_conv = 1; %plot the convergence of QCSIDCO algorithm
plot_MDFA = 1; %plot the miss detection and false alarm vs. SNR
plot_NMSE = 1; %plot the channel estimation error (NMSE) vs. SNR
plot_spar = 1; %plot the performance of sparsity

%% setup
% parameters
N = 200; %num. UEs
M = 55;  %len. spread sequence
K = 20;  %num. active UEs
J = 16;  %num. antennas of BS
beta  = 1;   %pathloss or shadowing components
SetUE = 1:N; %set of UEs

iter_sidco = 1e3; %num. of iterations for QCSIDCO algorithm (e.g. 1e3)

reals = 1e1; %num. channel realization (e.g. 1e6)

% variable
SNR = -10: 2 :10; %SNR[dB]
% SNR =0;
Noivar = 10.^(-SNR./10) / M; %noise variance (linear value)

% K-sparse vector
S         = zeros(N, 1);
S(1:K, :) = ones (K, 1);

if (frame_QCS)
    if (isfile('QCSIDCO.mat'))
        load('QCSIDCO.mat');
        [MM, NN] = size(A_QCS);
        if (MM ~= M || NN ~= N)
        msg = 'Error! Please generate the QCSIDCO frmae again!';
        error(msg);
        end
        [mcQCS, ~] = frameProperties(A_QCS);
        clear row column MM NN
    else
        addpath('QCSIDCO')
        %---for using cvx on the parallel server-------
        addpath('/app/MATLAB/cvx');
        cvx_setup;
        %--------
        % generate measurement matrix (M * N)
        % avgmc : column vector of average coherence
        % minmc : column vector of minimum coherence
        % maxmc : column vector of maximum coherence (mutual coherence of Hc)
        % stdmc : standard deviation of coherence
        [A_QCS, ~, avgmcQCS, minmcQCS, maxmcQCS, stdmcQCS] = generate_qcsidco_frames(iter_sidco, M, N);
        [row, column] = size(A_QCS);
        save('QCSIDCO.mat', 'A_QCS', 'avgmcQCS', 'minmcQCS', 'maxmcQCS', 'stdmcQCS', 'row', 'column', 'iter_sidco')
        clear row column
    end
end


%% main roop
% initialize
% mutual coherence
mcGRFreals = zeros(length(SNR), reals);
mcDFTreals = zeros(length(SNR), reals);
% SMV
% miss detection probability
pmdGRFreals = ones(length(SNR), reals);
pmdDFTreals = ones(length(SNR), reals);
pmdQCSreals = ones(length(SNR), reals);
% false alarm probability
pfaGRFreals = ones(length(SNR), reals) .* K/(N-K);
pfaDFTreals = ones(length(SNR), reals) .* K/(N-K);
pfaQCSreals = ones(length(SNR), reals) .* K/(N-K);
% NMSE
nmseORCreals = zeros(length(SNR), reals);
nmseGRFreals = zeros(length(SNR), reals);
nmseDFTreals = zeros(length(SNR), reals);
nmseQCSreals = zeros(length(SNR), reals);
% MMV
% miss detection probability
pmdGRFmmvreals = ones(length(SNR), reals);
pmdDFTmmvreals = ones(length(SNR), reals);
pmdQCSmmvreals = ones(length(SNR), reals);
% false alarm probability
pfaGRFmmvreals = ones(length(SNR), reals) .* K/(N-K);
pfaDFTmmvreals = ones(length(SNR), reals) .* K/(N-K);
pfaQCSmmvreals = ones(length(SNR), reals) .* K/(N-K);
% NMSE
nmseGRFmmvreals = zeros(length(SNR), reals);
nmseDFTmmvreals = zeros(length(SNR), reals);
nmseQCSmmvreals = zeros(length(SNR), reals);
% SNR
SNRreals = zeros(length(SNR), reals);

for sn = 1:length(SNR)
    nvar = Noivar(sn);
    for iter = 1:reals
        % active UE selection
        S_t = S(randperm(N));     %choose K active users randomly
        SetActiveUEs = find(S_t); %indeces of active UEs

        % channel generation
        h = sqrt(0.5) * (randn(N, J) + 1j * randn(N, J)) .* repmat(S_t, 1, J);
        
        % noise generation
        n = sqrt(0.50 * nvar) * (randn(M, J) + 1j * randn(M, J));
        
        % SNR
        SNRreals(sn, iter) = norm(h, 'fro')^2 / norm(n, 'fro')^2 / K; %average SNR of active UEs

        % AUD and CE by each frame
        % gaussian random frame
        if (frame_GRF)
            % generate the frame
            A_GRF = sqrt(0.5) * (randn(M, N) + 1j * randn(M, N));
            A_GRF = A_GRF./vecnorm(A_GRF, 2, 1); %normalization
            A = A_GRF; % for oracle mmse calculation

            % mutual coherence
            [mcGRFreals(sn, iter), ~] = frameProperties(A_GRF);

            % received signal
            y = A_GRF * h + n;
            
            % SMV : parallel processing per antenna
            if (flag_SMV)
                % OMP
                [xhat_GRF, SetEst_GRF] = OMP_nzknown(y, A_GRF, K);
                
                % MD, FA, AER
                for j = 1 : J
                    [pmd, pfa, ~] = Compute_MDandFA(SetActiveUEs, SetEst_GRF(:, j), N, K);
                end
                pmdGRFreals(sn, iter) = mean(pmd, 'all');
                pfaGRFreals(sn, iter) = mean(pfa, 'all');
    
                % MMSE
%                 nmseGRFreals(sn, iter) = ChannelEstimation_MMSE(A_GRF, y, h, xhat_GRF, SetEst_GRF, nvar);
                nmseGRFreals(sn, iter) = norm(h - xhat_GRF, 'fro')^2 / norm(h, 'fro')^2;
            end
            
            % MMV : simultaneous processing of all antennas
            if (flag_MMV)
                % OMP
                [xhat_GRFmmv, SetEst_GRFmmv] = MMVOMP_nzknown(y, A_GRF, K);

                % MD, FA, AER
                [pmdGRFmmvreals(sn, iter), pfaGRFmmvreals(sn, iter)] = Compute_MDandFA(SetActiveUEs, SetEst_GRFmmv, N, K);

                % MMSE
%                 nmseGRFmmvreals(sn, iter) = ChannelEstimation_MMSE(A_GRF, y, h, xhat_GRFmmv, SetEst_GRFmmv, nvar);
                nmseGRFmmvreals(sn, iter) = norm(h - xhat_GRFmmv, 'fro')^2 / norm(h, 'fro')^2;

            end
            
        end
        
        % partial DFT frame
        if (frame_DFT)
            % generate the frame
            dftmat = dftmtx(N) / sqrt(N);
            A_DFT = dftmat(randperm(N, M), :);
            A_DFT = A_DFT./vecnorm(A_DFT, 2, 1); %normalization
            A = A_DFT; %for oracle mmse calculation

            % mutual coherence
            [mcDFTreals(sn, iter), ~] = frameProperties(A_DFT);

            % received signal
            y = A_DFT * h + n;

            % SMV : parallel processing per antenna
            if (flag_SMV)
                % OMP
                [xhat_DFT, SetEst_DFT] = OMP_nzknown(y, A_DFT, K);
                
                % MD, FA, AER
                for j = 1 : J
                    [pmd, pfa, ~] = Compute_MDandFA(SetActiveUEs, SetEst_DFT(:, j), N, K);
                end
                pmdDFTreals(sn, iter) = mean(pmd, 'all');
                pfaDFTreals(sn, iter) = mean(pfa, 'all');
        
                % MMSE
%                 nmseDFTreals(sn, iter) = ChannelEstimation_MMSE(A_DFT, y, h, xhat_DFT, SetEst_DFT, nvar);
                nmseDFTreals(sn, iter) = norm(h - xhat_DFT, 'fro')^2 / norm(h, 'fro')^2;

            end
            
            % MMV : simultaneous processing of all antennas
            if (flag_MMV)
                % OMP
                [xhat_DFTmmv, SetEst_DFTmmv] = MMVOMP_nzknown(y, A_DFT, K);

                % MD, FA, AER
                [pmdDFTmmvreals(sn, iter), pfaDFTmmvreals(sn, iter)] = Compute_MDandFA(SetActiveUEs, SetEst_DFTmmv, N, K);

                % MMSE
%                 nmseDFTmmvreals(sn, iter) = ChannelEstimation_MMSE(A_DFT, y, h, xhat_DFTmmv, SetEst_DFTmmv, nvar);
                nmseDFTmmvreals(sn, iter) = norm(h - xhat_DFTmmv, 'fro')^2 / norm(h, 'fro')^2;

            end
            
        end

        % QCSIDCO frame
        if (frame_QCS)
            % frame is pre-generated
            A = A_QCS; %for oracle mmse calculation

            % received signal
            y = A_QCS * h + n;
            
            % SMV : parallel processing per antenna
            if (flag_SMV)
                % OMP
                [xhat_QCS, SetEst_QCS] = OMP_nzknown(y, A_QCS, K);
                
                % MD, FA, AER
                for j = 1 : J
                    [pmd, pfa, ~] = Compute_MDandFA(SetActiveUEs, SetEst_QCS(:, j), N, K);
                end
                pmdQCSreals(sn, iter) = mean(pmd, 'all');
                pfaQCSreals(sn, iter) = mean(pfa, 'all');
    
                % MMSE
%                 nmseQCSreals(sn, iter) = ChannelEstimation_MMSE(A_QCS, y, h, xhat_QCS, SetEst_QCS, nvar);
                nmseQCSreals(sn, iter) = norm(h - xhat_QCS, 'fro')^2 / norm(h, 'fro')^2;

            end
            
            % MMV : simultaneous processing of all antennas
            if (flag_MMV)
                % OMP
                [xhat_QCSmmv, SetEst_QCSmmv] = MMVOMP_nzknown(y, A_QCS, K);

                % MD, FA, AER
                [pmdQCSmmvreals(sn, iter), pfaQCSmmvreals(sn, iter)] = Compute_MDandFA(SetActiveUEs, SetEst_QCSmmv, N, K);

                % MMSE
%                 nmseQCSmmvreals(sn, iter) = ChannelEstimation_MMSE(A_QCS, y, h, xhat_QCSmmv, SetEst_QCSmmv, nvar);
                nmseQCSmmvreals(sn, iter) = norm(h - xhat_QCSmmv, 'fro')^2 / norm(h, 'fro')^2;

            end
        end
        
        % NMSE of the oracle MMSE
        nmseORCreals(sn, iter) = Oracle_MMSE(A, y, h, SetActiveUEs, nvar);

    end
end

[~, wb] = frameProperties(ones(M, N));

% calculate the expected values
% mutual coherence
% mcGRF = mean(mcGRFreals, 'all');
% mcDFT = mean(mcDFTreals, 'all');
% print('Mutual coherence of GRF')
% SMV
% miss detection probability
pmdGRF = mean(pmdGRFreals, 2);
pmdDFT = mean(pmdDFTreals, 2);
pmdQCS = mean(pmdQCSreals, 2);
% false alarm probability
pfaGRF = mean(pfaGRFreals, 2);
pfaDFT = mean(pfaDFTreals, 2);
pfaQCS = mean(pfaQCSreals, 2);
% NMSE
nmseORC = (mean(nmseORCreals, 2));
nmseGRF = (mean(nmseGRFreals, 2));
nmseDFT = (mean(nmseDFTreals, 2));
nmseQCS = (mean(nmseQCSreals, 2));
% MMV
% miss detection probability
pmdGRFmmv = mean(pmdGRFmmvreals, 2);
pmdDFTmmv = mean(pmdDFTmmvreals, 2);
pmdQCSmmv = mean(pmdQCSmmvreals, 2);
% false alarm probability
pfaGRFmmv = mean(pfaGRFmmvreals, 2);
pfaDFTmmv = mean(pfaDFTmmvreals, 2);
pfaQCSmmv = mean(pfaQCSmmvreals, 2);
% NMSE
nmseGRFmmv = (mean(nmseGRFmmvreals, 2));
nmseDFTmmv = (mean(nmseDFTmmvreals, 2));
nmseQCSmmv = (mean(nmseQCSmmvreals, 2));
% SNR
SNRtrue = 10*log10(mean(SNRreals, 2));


%% plot
% plot the convergence properties
if (plot_conv)
    f1 = figure(1);
    % f1.Position(3:4) = [900 450]; % for draft
    f1.Position(3:4) = [560 420]; % for slide
    % f1.Position(3:4) = [600 350]; % for thesis
    hold on; grid on; box on;

    optimizationtimes = length(maxmcQCS);
    [mcQCS, wb] = frameProperties(A_QCS);

    wbline  = plot(1:optimizationtimes, repmat(wb, [1 optimizationtimes]), 'k:', 'Linewidth', 2);
    qcsconv = plot(1:optimizationtimes, maxmcQCS, '--', 'LineWidth',2);
    qcsconv.Color = genRGBForPlot(2);
    xlabel("Num. Iterations" , "Fontsize", 15, "Fontname", "Times New Roman");
    ylabel("Mutual Coherence", "Fontsize", 15, "Fontname", "Times New Roman");
    title('Convergence of the QCSIDCO', 'Interpreter', 'Latex', 'Fontsize', 14);
    legend([qcsconv, wbline], {'QCSIDCO', 'Welch Bound'}, 'Interpreter', 'Latex', 'Location', 'northeast', 'Fontsize', 15);
    xlim([1 optimizationtimes])
%         xticks(1:optimizationtimes)
end

% false alarm and miss detection
if (plot_MDFA)
f2 = figure(2);
    % f2.Position(3:4) = [900 450]; % for draft
    f2.Position(3:4) = [560 420]; % for slide
    % f2.Position(3:4) = [600 350]; % for thesis
    
%     pmd_grf = semilogy(SNR, pmdGRF, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
%     hold on
    pmd_dft = semilogy(SNR, pmdDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    hold on;
%     pmd_qcs = semilogy(SNR, pmdQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');

%     pmd_grfmmv = semilogy(SNR, pmdGRFmmv, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    pmd_dftmmv = semilogy(SNR, pmdDFTmmv, '-d', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
%     pmd_qcsmmv = semilogy(SNR, pmdQCSmmv, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');

%     pfa_grf = semilogy(SNR, pfaGRF, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');
    pfa_dft = semilogy(SNR, pfaDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');
%     pfa_qcs = semilogy(SNR, pfaQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');

%     pfa_grfmmv = semilogy(SNR, pfaGRFmmv, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');
    pfa_dftmmv = semilogy(SNR, pfaDFTmmv, '-d', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');
%     pfa_qcsmmv = semilogy(SNR, pfaQCSmmv, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');
    
    pmd_dftmmv.Color = genRGBForPlot(4);
    pmd_dftmmv.MarkerFaceColor = genRGBForPlot(4);
    pfa_dftmmv.Color = genRGBForPlot(4);

    ylim([1e-5 1e-0])
    grid on
    box on
    xlabel("SNR [dB]" , "Fontsize", 15, "Fontname", "Times New Roman");
    ylabel("Probability", "Fontsize", 15, "Fontname", "Times New Roman");
    title('False Alarm and Miss Detection', 'Interpreter', 'Latex', 'Fontsize', 14);
%     legend([pmd_grf, pmd_dft, pmd_qcs, pmd_grfmmv, pmd_dftmmv, pmd_qcsmmv, pfa_grf, pfa_dft, pfa_qcs, pfa_grfmmv, pfa_dftmmv, pfa_qcsmmv],...
%         {'Gaussian (MD)', 'Partial DFT (MD)', 'QCSIDCO (MD)', 'Gaussian (MD)', 'Partial DFT (MD)', 'QCSIDCO (MD)', 'Gaussian (FA)','Partial DFT (FA)', 'QCSIDCO (FA)', 'Gaussian (FA)','Partial DFT (FA)', 'QCSIDCO (FA)'},...
%         'Interpreter', 'Latex', 'Location', 'southwest', 'Fontsize', 10);
    legend([pmd_dft, pmd_dftmmv, pfa_dft, pfa_dftmmv], {'Partial DFT (MD, P-MMV)','Partial DFT (MD, MMV-OMP)', 'Partial DFT (FA, P-OMP)', 'Partial DFT (FA, MMV-OMP)'}, 'Interpreter', 'Latex', 'Location', 'northeast', 'Fontsize', 13);

end

% NMSE
if (plot_NMSE)
    f3 = figure(3);
    % f3.Position(3:4) = [900 450]; % for draft
    f3.Position(3:4) = [560 420]; % for slide
    % f3.Position(3:4) = [600 350]; % for thesis
    
    pnm_orc = semilogy(SNR, nmseORC, 'k:', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on
%     pnm_grf = semilogy(SNR, nmseGRF, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    pnm_dft = semilogy(SNR, nmseDFT, 'b-.^', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
%     pnm_qcs = semilogy(SNR, nmseQCS, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
%     pnm_grfmmv = semilogy(SNR, nmseGRFmmv, 'k--s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');
    pnm_dftmmv = semilogy(SNR, nmseDFTmmv, 'b-d', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');
%     pnm_qcsmmv = semilogy(SNR, nmseQCSmmv, 'r-o' , 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'w');

    pnm_dftmmv.Color = genRGBForPlot(4);
    pnm_dftmmv.MarkerFaceColor = genRGBForPlot(4);
    

    xlabel("SNR [dB]" , "Fontsize", 15, "Fontname", "Times New Roman");
    ylabel("NMSE", "Fontsize", 15, "Fontname", "Times New Roman");
    title('NMSE performances', 'Interpreter', 'Latex', 'Fontsize', 14);
    ylim([2e-3 1e0])
    grid on
    box on
%     legend([pnm_grf, pnm_dft, pnm_qcs, pnm_grfmmv, pnm_dftmmv, pnm_qcsmmv, pnm_orc], {'Gaussian', 'Partial DFT', 'QCSIDCO', 'Gaussian (MMV)', 'Partial DFT (MMV)', 'QCSIDCO (MMV)', 'Oracle MMSE'}, 'Interpreter', 'Latex', 'Location', 'southwest', 'Fontsize', 13);
    legend([pnm_dft, pnm_dftmmv, pnm_orc], {'Partial DFT (P-OMP)', 'Partial DFT (MMV-OMP)', 'Oracle MMSE'}, 'Interpreter', 'Latex', 'Location', 'northeast', 'Fontsize', 15);

end

% sparsity
if (plot_spar)
    f4 = figure(4);
    tiledlayout(1, 3)
    % tile 1
    nexttile
    spy(h);
    title('Original')
    % tile 2
    nexttile
    spy(xhat_DFT);
    title('Reconst. w/ P-OMP')
    % tile 3
    nexttile
    spy(xhat_DFTmmv);
    title('Reconst. w/ MMV-OMP')

end



toc