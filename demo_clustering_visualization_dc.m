clear
close all

rng('twister')


addpath(genpath('algs'))
addpath(genpath('utils'))


% setting path
data_path = 'data/doc';

data_set = {
    'bbcsport.mat',
};

test_num = 1;

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



for t = test_num
    load(fullfile(data_path, data_set{t}))
    nClass = length(unique(gnd));
    fea0 = fea;
    fea = NormalizeFea(fea);
    A = fea';
    fprintf('Running on the dataset: %s......\n', data_set{t});

    % Recorder
    alg_name = cell(algLength, 1);
    alg_cpu = cell(algLength, 1);
    alg_purity = cell(algLength, 1);
    alg_nmi = cell(algLength, 1);
    alg_MIhat = cell(algLength, 1);
    alg_AC = cell(algLength, 1);

    ret_label = cell(algLength, 1);

    for nn = 1:numRuns
        alg_cnt = 1;
        

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
            [sol, ~] = dtpp_nmf(A, nClass, options); 
            [~, label] = max(abs(sol.H'), [], 2);
            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label_permute;
            alg_cnt = alg_cnt + 1;
        end

        % MU-ONMF
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
            [sol, ~] = orth_mu_nmf(A, nClass, options); 

            [~, label] = max(abs(sol.H'), [], 2);
            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label_permute;
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
            [sol, ~] = hals_so_nmf(A, nClass, options);

            [~, label] = max(abs(sol.H'), [], 2);

            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label_permute;
            alg_cnt = alg_cnt + 1;
        end

        % EM-ONMF
        if flag_alg('EM-ONMF')
            alg_name{alg_cnt} = 'EM-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;

            numClusters = nClass;
            maxEmIters = 100;
            [clusters_emonmf,Vemonmf,relError,actualIters] = emonmf(A',numClusters,maxEmIters);

            label = clusters_emonmf;
            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label_permute;
            alg_cnt = alg_cnt + 1;
        end

        % ONPMF 
        if flag_alg('ONPMF')
            alg_name{alg_cnt} = 'ONPMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            numClusters = nClass;
            maxOnpmfIters = 3000;
            [Uonpmf,Vonpmf,relError,actualIters] = onpmf(A',numClusters,maxOnpmfIters);

            [~, label] = max(abs(Uonpmf), [], 2);
            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label_permute;
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
            opts.r = nClass;
            [Kc, Bc, Out] = sn_onmf(A', opts);


            [~, label] = max(abs(Bc), [], 2);

            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label_permute;
            alg_cnt = alg_cnt + 1;
        end

        % NS-ONMF - Proposed
        if flag_alg('NS-ONMF')
            alg_name{alg_cnt} = 'NS-ONMF';
            fprintf('Processing alg: %12s\n', alg_name{alg_cnt})
            cpu0 = tic;
            opts = [];
            opts.max_iter = 1000;
            opts.BCD_MIter = 1;
            opts.epsilon = 1e-8;
            opts.flag_debug = 1;
            opts.r = nClass;
            opts.rho = 1e-6;
            opts.rhomax = 1e10;
            opts.eta = 0.99;
            opts.gamma = 1.1;
            opts.tau = 0.1;


            [Zc, Out] = NS_ONMF(A, opts);
            [~, label] = max(abs(Out.C), [], 2);


            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label_permute = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label_permute);
            AC = length(find(gnd == label_permute))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label_permute;
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
                alg_name{j}, mean(alg_cpu{j}), mean(alg_purity{j}), mean(alg_nmi{j}), ...
                    mean(alg_MIhat{j}), mean(alg_AC{j}));
        end
    end

    %% Visual Plot
    flag_visual_plot = 1;
    if flag_visual_plot
        fea0 = full(fea0);
        XY = tsne(full(fea0)); 

        term_path = fullfile(data_path, 'bbcsport/bbcsport.terms');

        fid = fopen(term_path, 'r');
        terms =  textscan(fid,'%s', 'Delimiter', '\n');
        fclose(fid);
        uniqueWords = terms{:};


        k = 3;
        topWords = strings(nClass, 1);
        for i = 1:nClass
            counts = fea0(gnd==i, :);
            bag = bagOfWords(uniqueWords,counts);
            top = topkwords(bag, k);
            topWords(i) = join(top.Word,", ");
        end

        % figure('DefaultTextFontSize',20, 'DefaultAxesFontSize', 20);
        figure
        tiledlayout(2,4, 'TileSpacing', 'Compact')

        nexttile
        gscatter(XY(:,1), XY(:, 2), gnd);
        title("Ground Truth")
        legend(topWords, Location="southoutside", NumColumns=2)

        for j = 1:alg_cnt-1
            nexttile
            gscatter(XY(:,1), XY(:, 2), ret_label{j});
            title(alg_name{j})
            legend(topWords, Location="southoutside", NumColumns=2)
        end
    end


end
