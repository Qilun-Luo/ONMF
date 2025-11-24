% Check for convergence
clear
close all

rng('twister')

addpath(genpath('algs'))
addpath(genpath('utils'))

numRuns = 30;

% Algs setting
flag_alg = dictionary;
flag_alg('NS-ONMF') = 1; % Proposed

algLength = sum(values(flag_alg));

caseNum = 3;
sigma = 1e-4;


for cn = caseNum

    % Recorder
    alg_name = cell(algLength, 1);
    alg_cpu = cell(algLength, 1);
    alg_deltaM = cell(algLength, 1);
    alg_exactbnd = cell(algLength, 1);


    % main runs
    for t = 1:numRuns


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

        alg_cnt = 1;
        
        % Algs
        fprintf('Run #%d for case %d...\n', t, cn);
    
        if flag_alg('NS-ONMF')
            alg_name{alg_cnt} = 'NS-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            opts = [];
            opts.max_iter = 1000;
            opts.BCD_MIter = 10;
            opts.epsilon = 1e-8;
            opts.flag_debug = 0;
            opts.r = k;
            opts.rho = 1e-10;
            opts.rhomax = 1e8;
            opts.eta = 0.999;
            opts.gamma = 1.2;
            opts.tau = 0.1;

            [Zc, Out_NS_ONMF] = NS_ONMF(A, opts);
            Cc = Out_NS_ONMF.C;
            Bc = Out_NS_ONMF.B;
            Kc = Out_NS_ONMF.K;
            Ac = Bc*Cc';
            Mc = Zc*Zc';
            M = eye(n) - C*C';


            % --- Record ---
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_deltaM{alg_cnt} = [alg_deltaM{alg_cnt}; norm(Mc-M, 'fro')];
            alg_exactbnd{alg_cnt} = [alg_exactbnd{alg_cnt}; 2*sqrt(2)*norm(C'*pinv(A_gt), 2)*norm((A_gt-A), 'fro')];
            alg_cnt  = alg_cnt + 1;
        end


    end


    flag_plot = 1;
    if flag_plot
        for j = 1:alg_cnt-1
            f = figure;
            
            plot(alg_deltaM{j}, 'Marker', 'o', 'LineWidth', 2, 'MarkerSize',10);
            hold on;
            plot(alg_exactbnd{j}, 'Marker', '>', 'LineWidth', 2, 'MarkerSize',10);
            xlabel('Runs', 'Fontsize', 20)
            ylabel('Bounds', 'Fontsize', 20);
            hold off
            legend({'$\|\Delta M\|_F$', 'Bound: $2\sqrt{2}\|C^TA^\dag\|_2 \|\Delta A\|_F$'}, 'interpreter', 'latex', 'Fontsize', 20)
            set(gca,'YScale','log')
            exportgraphics(f, 'output/errorbounds.png')
        end
    end

end