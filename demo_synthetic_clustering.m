% demo for more synthetic data
clear
close all

rng('twister')

addpath(genpath('algs'))
addpath(genpath('utils'))


numRuns = 1;

% Algs setting
flag_alg = dictionary;
flag_alg('BiOR-NM3F') = 1;
flag_alg('MU-ONMF') = 1;
flag_alg('HALS') = 1;
flag_alg('EM-ONMF') = 1;
flag_alg('ONPMF') = 1;
flag_alg('SN-ONMF') = 1;
flag_alg('NS-ONMF') = 1; % Proposed
algLength = sum(values(flag_alg));


epsList = logspace(-2, 0, 20);

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


         % BiOR-NM3F
         if flag_alg('BiOR-NM3F')
            alg_name{alg_cnt} = 'BiOR-NM3F';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            options = [];
            options.not_store_infos = true;
            options.verbose = 0;
            options.max_epoch = 2000; 
            options.orth_h    = 1;
            options.norm_h    = 2;
            options.orth_w    = 0;
            options.norm_w    = 0; 
            [sol, ~] = dtpp_nmf(A, K, options); 
            [~, label] = max(abs(sol.H'), [], 2);
            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
             % Record
             alg_cpu{alg_cnt, ee} = [alg_cpu{alg_cnt, ee}; toc(cpu0)];
             alg_purity{alg_cnt, ee} = [alg_purity{alg_cnt, ee}; purity];
             alg_nmi{alg_cnt, ee} = [alg_nmi{alg_cnt, ee}; nmi];
             alg_MIhat{alg_cnt, ee} = [alg_MIhat{alg_cnt, ee};  MIhat];
             alg_AC{alg_cnt, ee} = [alg_AC{alg_cnt, ee}; AC];
             alg_cnt = alg_cnt + 1;
        end


        % Orth-MU
        if flag_alg('MU-ONMF')
            alg_name{alg_cnt} = 'MU-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            options = [];
            options.max_epoch = 2000;
            options.verbose = 0;
            options.not_store_infos = true;
            options.orth_h    = 1;
            options.norm_h    = 2;
            options.orth_w    = 0;
            options.norm_w    = 0;    
            [sol, ~] = orth_mu_nmf(A, K, options); 

            [~, label] = max(abs(sol.H'), [], 2);
            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
             % Record
             alg_cpu{alg_cnt, ee} = [alg_cpu{alg_cnt, ee}; toc(cpu0)];
             alg_purity{alg_cnt, ee} = [alg_purity{alg_cnt, ee}; purity];
             alg_nmi{alg_cnt, ee} = [alg_nmi{alg_cnt, ee}; nmi];
             alg_MIhat{alg_cnt, ee} = [alg_MIhat{alg_cnt, ee};  MIhat];
             alg_AC{alg_cnt, ee} = [alg_AC{alg_cnt, ee}; AC];
             alg_cnt = alg_cnt + 1;
        end

        % HALS
        if flag_alg('HALS')
            alg_name{alg_cnt} = 'HALS';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            options = [];
            options.not_store_infos = true;
            options.verbose = 0;
            options.max_epoch = 2000; 
            options.wo = 1;
            [sol, ~] = hals_so_nmf(A, K, options);

            [~, label] = max(abs(sol.H'), [], 2);

            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
              % Record
            alg_cpu{alg_cnt, ee} = [alg_cpu{alg_cnt, ee}; toc(cpu0)];
            alg_purity{alg_cnt, ee} = [alg_purity{alg_cnt, ee}; purity];
            alg_nmi{alg_cnt, ee} = [alg_nmi{alg_cnt, ee}; nmi];
            alg_MIhat{alg_cnt, ee} = [alg_MIhat{alg_cnt, ee};  MIhat];
            alg_AC{alg_cnt, ee} = [alg_AC{alg_cnt, ee}; AC];
            alg_cnt = alg_cnt + 1;
        end

        
       
        % EM-ONMF
        if flag_alg('EM-ONMF')
            alg_name{alg_cnt} = 'EM-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;

            numClusters = K;
            maxEmIters = 100;
            [clusters_emonmf,Vemonmf,relError,actualIters] = emonmf(A',numClusters,maxEmIters);

            label = clusters_emonmf;
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

        % ONPMF 
        if flag_alg('ONPMF')
            alg_name{alg_cnt} = 'ONPMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            numClusters = K;
            maxOnpmfIters = 3000;
            [Uonpmf,Vonpmf,relError,actualIters] = onpmf(A',numClusters,maxOnpmfIters);

            [~, label] = max(abs(Uonpmf), [], 2);
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

        % SN-ONMF
        if flag_alg('SN-ONMF')
            alg_name{alg_cnt} = 'SN-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            opts = [];
            opts.max_iter = 100;
            opts.epsilon = 1e-6;
            opts.flag_debug = 0;
            opts.theta = 1e-6;
            opts.beta = 1e-7;
            opts.mu1 = 5e-1;
            opts.mu2 = 5e-1;
            opts.mu3 = 5e-1;
            opts.eta = 1.1;
            opts.r = K;
            [Kc, Bc, Out] = sn_onmf(A', opts);

            [~, label] = max(abs(Bc), [], 2);

            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
             % Record
             alg_cpu{alg_cnt, ee} = [alg_cpu{alg_cnt, ee}; toc(cpu0)];
             alg_purity{alg_cnt, ee} = [alg_purity{alg_cnt, ee}; purity];
             alg_nmi{alg_cnt, ee} = [alg_nmi{alg_cnt, ee}; nmi];
             alg_MIhat{alg_cnt, ee} = [alg_MIhat{alg_cnt, ee};  MIhat];
             alg_AC{alg_cnt, ee} = [alg_AC{alg_cnt, ee}; AC];
             alg_cnt = alg_cnt + 1;
        end

        % NS-ONMF - Proposed
        if flag_alg('NS-ONMF')
            alg_name{alg_cnt} = 'NS-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            opts = [];
            opts.max_iter = 1000;
            opts.BCD_MIter = 10;
            opts.epsilon = 1e-6;
            opts.rhomax = 1e8;
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
end




flag_plot = 1;
if flag_plot
    mean_cpu = cellfun(@mean,alg_cpu);
    mean_purity = cellfun(@mean, alg_purity);
    mean_nmi = cellfun(@mean, alg_nmi);
    mean_MIhat = cellfun(@mean, alg_MIhat);
    mean_AC = cellfun(@mean, alg_AC);

    h_AC = figure('DefaultAxesFontSize', 16);
    semilogx(epsList, mean_AC', 'LineWidth', 2, 'MarkerSize', 10)
    ax = gca;
    mylinestyles = ["-*"; "-square"; "-o"; "-diamond"; "-^"; "-v";"->"; "-<"; "-pentagram"];
    ax.LineStyleOrder = mylinestyles;
    ax.LineStyleCyclingMethod = 'withcolor';
    xlabel('Noise level $\varepsilon$', 'Interpreter', 'latex')
    ylabel('Accuracy')
    xlim([min(epsList), max(epsList)])
    legend(alg_name, 'Location','SouthWest')
end


