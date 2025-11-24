% Check for convergence
clear
close all

rng('twister')

addpath(genpath('algs'))
addpath(genpath('utils'))


numRuns = 1;

% Algs setting
flag_alg = dictionary;
flag_alg('NS-ONMF') = 1; % Proposed
algLength = sum(values(flag_alg));

epsList = [0.1];
% epsList = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1];

epsLength = length(epsList);

% Recorder
alg_name = cell(algLength, 1);
alg_cpu = cell(algLength, epsLength);
alg_purity = cell(algLength, epsLength);
alg_nmi = cell(algLength, epsLength);
alg_MIhat = cell(algLength, epsLength);
alg_AC = cell(algLength, epsLength);

for ee = 1:epsLength % noise level
    epsilon = epsList(ee);


    % Data generating
    m = 1000;
    K = 6;
    numClass = 100 - ((1:K)-1)*10;

    centroid = zeros(m, K);
    for i = 1:K
        centroid(:,i) = rand(m, 1);
    end
    n = sum(numClass);

    B = centroid;
    A = zeros(m, n);
    gnd = zeros(n, 1);
    C = zeros(n, K);
    s = 0;
    for k = 1:K
        for i = 1:numClass(k)
            j = s + i;
            alpha = 0.8*rand + 0.2;
            A(:, j) = alpha*centroid(:,k) + rand(m, 1)*epsilon;
            gnd(j) = k;
            C(j, k) = alpha;
        end
        s = s + numClass(k);
    end
    A(A<0) = 0;

    ind = randperm(n);
    % ind = 1:n;
    gnd = gnd(ind);
    A = A(:, ind);
    C = C(ind, :);

    B = B.*vecnorm(C);
    C = C./vecnorm(C);
    A0 = B*C';
    % fprintf('||A-A0||_F: %.12e\n', norm(A-A0, 'fro'))




    for t = 1:numRuns
        
        alg_cnt = 1;
        
        % Algs
        fprintf('Run #%d/%d with the noise level: %.9f\n', t, numRuns, epsilon);

        % NS-ONMF - Proposed
        if flag_alg('NS-ONMF')
            alg_name{alg_cnt} = 'NS-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            opts = [];
            opts.max_iter = 1000;
            opts.BCD_MIter = 10;
            opts.epsilon = 1e-4;
            opts.rhomax = 1e10;
            opts.flag_debug = 1;
            opts.r = K;
            opts.rho = 1e-2;
            opts.eta = 0.99;
            opts.gamma = 1.5;
            opts.tau = 0.1;


            [Zc, Out] = NS_ONMF(A, opts);


            [~, label] = max(abs(Out.C), [], 2);


            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label);
            AC = length(find(gnd == label))/length(gnd);
            % Record
            alg_cpu{alg_cnt, ee} = [alg_cpu{alg_cnt, ee}; toc(cpu0)];
            alg_purity{alg_cnt, ee} = [alg_purity{alg_cnt, ee}; purity];
            alg_nmi{alg_cnt, ee} = [alg_nmi{alg_cnt, ee}; nmi];
            alg_MIhat{alg_cnt, ee} = [alg_MIhat{alg_cnt, ee};  MIhat];
            alg_AC{alg_cnt, ee} = [alg_AC{alg_cnt, ee}; AC];
            alg_cnt = alg_cnt + 1;
        end


    end
    %% Table reporting
    flag_report = 1;
    if flag_report
        fprintf('%12s\t%4s\t%4s\t%4s\t%4s\t%4s\n', 'Algs', 'CPU', ...
        'Purity', 'NMI', 'MIhat', 'AC');
        for j = 1:alg_cnt-1
            fprintf('%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
                alg_name{j}, mean(alg_cpu{j, ee}), mean(alg_purity{j, ee}), mean(alg_nmi{j, ee}), ...
                    mean(alg_MIhat{j, ee}), mean(alg_AC{j, ee}));
        end
    end

    %% Plot
    flag_plot = 1;
    if flag_plot
        f1 = figure;
        hold on;
        semilogy(Out.obj, 'LineWidth', 2);
        xlabel('Iteration','fontsize',20)
        ylabel('$\mathcal{L}(\mathcal{W}^k, \Lambda^k, \rho_k)$', 'Interpreter', 'latex','fontsize',20)
        hold off
        exportgraphics(f1, 'output/conv_obj.png')
        


        f2 = figure;
        plot(Out.nrmC, 'LineWidth', 2);
        xlabel('Iteration','fontsize',20)
        ylabel('$\|c(\mathcal{W}^k)\|_F$', 'Interpreter', 'latex','fontsize',20)
        exportgraphics(f2, 'output/conv_cW.png')
    end
end

