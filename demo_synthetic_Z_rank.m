clear
close all

rng("twister")

addpath(genpath('algs'))
addpath(genpath('utils'))

numRuns = 10;

% Algs setting
flag_alg = dictionary;
flag_alg('BiOR-NM3F') = 1;
flag_alg('MU-ONMF') = 1;
flag_alg('EM-ONMF') = 1;
flag_alg('ONPMF') = 1;
flag_alg('SN-ONMF') = 1;
flag_alg('NS-ONMF') = 1; % Proposed

algLength = sum(values(flag_alg));

caseNum = 3;
rankList = 1:20;

rseList = zeros(algLength, length(rankList));


for cn = caseNum

    % Data generation
    m = 300;
    k = 10;
    n = 100;

    min_v = 5;
    points = sort(randi([1, n-k*min_v-1], 1, k-1));
    parts  = diff([0, points, n-k*min_v]);
    parts  = parts(randperm(length(parts ))) + min_v;
    points = [0 cumsum(parts)];
    C = zeros(n, k);
    for i = 1:k
        C(points(i)+1:points(i+1), i) = rand(parts(i), 1);
    end
    C = C./vecnorm(C);
    B = rand(m, k);

    for r_k = 1:length(rankList)
        select_k = rankList(r_k);

        switch cn
            case 1
                A = B*C';
                K = C*C';
                A_gt = A;
            case 2
                ind = randperm(n);
                C = C(ind, :);
                A = B*C';
                K = C*C';
                A_gt = A;
            case 3
                sigma = 1e-6;
                E = sigma*rand(m, n);
                A = B*C';
                K = C*C';
                A_gt = A;
                A = A + E;
        end


        % Compute the groundtruth of Z
        [VV, DD] = eig(C*C');
        [~, sorted_indices] = sort(diag(DD));
        Z = VV(:, sorted_indices(1:n-k));

        fprintf('||AZ||_F = %.12f\n', norm(A*Z, 'fro'));

        % Recorder
        alg_name = cell(algLength, 1);
        alg_cpu = cell(algLength, 1);
        alg_errC = cell(algLength, 1);
        alg_orthC = cell(algLength, 1);
        alg_resC = cell(algLength, 1);

        % main runs
        for t = 1:numRuns
            alg_cnt = 1;
            
            % Algs
            fprintf('Run #%d for case %d...\n', t, cn);

            if flag_alg('BiOR-NM3F')
                alg_name{alg_cnt} = 'BiOR-NM3F';
                fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
                cpu0 = tic;
                options = [];
                options.max_epoch = 10000;
                options.verbose = 0;
                options.not_store_infos = true;
                options.orth_h    = 1;
                options.norm_h    = 1;
                options.orth_w    = 0;
                options.norm_w    = 0;
                [sol, ~] = dtpp_nmf(A, select_k, options);

                Cc = sol.H';
                Bc = sol.W;

                % --- Record ---
                alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
                alg_errC{alg_cnt} = [alg_errC{alg_cnt}; norm(Cc*Cc'-C*C', 'fro')];
                alg_orthC{alg_cnt} = [alg_orthC{alg_cnt}; norm(Cc'*Cc - eye(select_k), 'fro')];
                alg_resC{alg_cnt} = [alg_resC{alg_cnt}; norm(A_gt-Bc*Cc', 'fro')/norm(A_gt, 'fro')];
                alg_cnt  = alg_cnt + 1;
            end

            if flag_alg('MU-ONMF')
                alg_name{alg_cnt} = 'MU-ONMF';
                fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
                cpu0 = tic;
                options = [];
                options.max_epoch = 10000;
                options.verbose = 0;
                options.not_store_infos = true;
                options.orth_h    = 1;
                options.norm_h    = 2;
                options.orth_w    = 0;
                options.norm_w    = 0;
                [sol, ~] = orth_mu_nmf(A, select_k, options); 
                Cc = sol.H';
                Bc = sol.W;

                % --- Record ---
                alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
                alg_errC{alg_cnt} = [alg_errC{alg_cnt}; norm(Cc*Cc'-C*C', 'fro')];
                alg_orthC{alg_cnt} = [alg_orthC{alg_cnt}; norm(Cc'*Cc - eye(select_k), 'fro')];
                alg_resC{alg_cnt} = [alg_resC{alg_cnt}; norm(A_gt-Bc*Cc', 'fro')/norm(A_gt, 'fro')];
                alg_cnt  = alg_cnt + 1;
            end

            if flag_alg('EM-ONMF')
                alg_name{alg_cnt} = 'EM-ONMF';
                fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
                cpu0 = tic;

                numClusters = select_k;
                maxEmIters = 100;
                [clusters,V_emonmf,relError,actualIters,U_binarized] = emonmf(A',numClusters,maxEmIters);
                W = V_emonmf';
                H = (A'*W).*U_binarized;
                Bc = W.*vecnorm(H);
                Cc = H./vecnorm(H);

                % --- Record ---
                alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
                alg_errC{alg_cnt} = [alg_errC{alg_cnt}; norm(Cc*Cc'-C*C', 'fro')];
                alg_orthC{alg_cnt} = [alg_orthC{alg_cnt}; norm(Cc'*Cc - eye(select_k), 'fro')];
                alg_resC{alg_cnt} = [alg_resC{alg_cnt}; norm(A_gt-Bc*Cc', 'fro')/norm(A_gt, 'fro')];
                alg_cnt  = alg_cnt + 1;
            end

            if flag_alg('ONPMF') 
                alg_name{alg_cnt} = 'ONPMF';
                fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
                cpu0 = tic;

                numClusters = select_k;
                maxOnpmfIters = 3000;
                [Uonpmf,V,relError,actualIters] = onpmf(A',numClusters,maxOnpmfIters);

                Cc = Uonpmf;
                Bc = V';

                % --- Record ---
                alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
                alg_errC{alg_cnt} = [alg_errC{alg_cnt}; norm(Cc*Cc'-C*C', 'fro')];
                alg_orthC{alg_cnt} = [alg_orthC{alg_cnt}; norm(Cc'*Cc - eye(select_k), 'fro')];
                alg_resC{alg_cnt} = [alg_resC{alg_cnt}; norm(A_gt-Bc*Cc', 'fro')/norm(A_gt, 'fro')];
                alg_cnt  = alg_cnt + 1;
            end

            if flag_alg('SN-ONMF')
                alg_name{alg_cnt} = 'SN-ONMF';
                fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
                cpu0 = tic;
                opts = [];
                opts.max_iter = 10000;
                opts.epsilon = 1e-6;
                opts.flag_debug = 0;
                opts.theta = 1e-6;
                opts.beta = 1e-7;
                opts.mu1 = 1e-1;
                opts.mu2 = 1e-1;
                opts.mu3 = 1e-1;
                opts.eta = 1.05;
                opts.r = select_k;
                [Kc, Cc, Out_SN_ONMF] = sn_onmf(A', opts);

                Bc = A*Cc;

                % --- Record ---
                alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
                alg_errC{alg_cnt} = [alg_errC{alg_cnt}; norm(Cc*Cc'-C*C', 'fro')];
                alg_orthC{alg_cnt} = [alg_orthC{alg_cnt}; norm(Cc'*Cc - eye(select_k), 'fro')];
                alg_resC{alg_cnt} = [alg_resC{alg_cnt}; norm(A_gt-Bc*Cc', 'fro')/norm(A_gt, 'fro')];
                alg_cnt  = alg_cnt + 1;
            end

            if flag_alg('NS-ONMF')
                alg_name{alg_cnt} = 'NS-ONMF';
                fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
                cpu0 = tic;
                opts = [];
                opts.max_iter = 1000;
                opts.BCD_MIter = 10;
                opts.epsilon = 1e-8;
                opts.flag_debug = 0;
                opts.r = select_k;
                opts.rho = 1e-10;
                opts.rhomax = 1e8;
                opts.eta = 0.999;
                opts.gamma = 1.2;
                opts.tau = 0.1;

                [Zc, Out_NS_ONMF] = NS_ONMF(A, opts);
                Cc = Out_NS_ONMF.C;
                Bc = Out_NS_ONMF.B;


                % --- Record ---
                alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
                alg_errC{alg_cnt} = [alg_errC{alg_cnt}; norm(Cc*Cc'-C*C', 'fro')];
                alg_orthC{alg_cnt} = [alg_orthC{alg_cnt}; norm(Cc'*Cc - eye(select_k), 'fro')];
                alg_resC{alg_cnt} = [alg_resC{alg_cnt}; norm(A_gt-Bc*Cc', 'fro')/norm(A_gt, 'fro')];
                alg_cnt  = alg_cnt + 1;
            end
        end
    

        % Table reporting
        mean_cpu = cellfun(@mean, alg_cpu);
        mean_errC = cellfun(@mean, alg_errC);
        mean_orthC = cellfun(@mean, alg_orthC);
        mean_resC = cellfun(@mean, alg_resC);

        flag_report = 1;
        if flag_report
            fprintf('%12s\t%8s\t%8s\t%8s\t%8s\n', 'Algs', 'CPU', 'errC', 'orthC', 'resC');
            for j = 1:alg_cnt-1
                fprintf('%12s\t%.4f\t%.8e\t%.8e\t%.8e\n', ...
                    alg_name{j}, mean_cpu(j), mean_errC(j), mean_orthC(j), mean_resC(j));
            end
        end

        rseList(:, r_k) = mean_resC;
    end

    flag_plot = 1;
    flag_plot_save = 1;
    if flag_plot
        f = figure;
        fontsize(16,"points")
        % f= figure('units','normalized','outerposition',[0 0 1 1]);
        semilogy(rankList, rseList', '-o', "LineWidth", 2)
        legend(alg_name, 'Location', 'southeast')
        xlim([min(rankList), max(rankList)]);
        xticks(rankList)
        xticklabels(rankList)
        xlabel('factor $r$', 'Fontsize', 16, 'Interpreter', 'latex')
        ylabel('res($\tilde{B},\tilde{C}$)', 'Fontsize', 16, 'Interpreter', 'latex');
        if flag_plot_save
            set(gcf, 'WindowState', 'maximized');
            exportgraphics(gca, 'output/rank_vs_res_10.png')
        end
    end
end