clear
close all

rng('twister')

addpath(genpath('algs'))
addpath(genpath('utils'))


% setting path
data_path = 'data/face';

data_set = {
    'PIE_pose27.mat',
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

% Recorder
alg_name = cell(algLength, 1);
alg_cpu = cell(algLength, 1);
alg_purity = cell(algLength, 1);
alg_nmi = cell(algLength, 1);
alg_MIhat = cell(algLength, 1);
alg_AC = cell(algLength, 1);

ret_label = cell(algLength, 1);
ret_basis = cell(algLength, 1);

for t = test_num
    load(fullfile(data_path, data_set{t}))
    nClass = length(unique(gnd));
    fea0 = fea;
    fea = NormalizeFea(fea);
    A = fea';
    fprintf('Running on the dataset: %s......\n', data_set{t});

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
            ret_label{alg_cnt} = label;
            ret_basis{alg_cnt} = sol.W;
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
            ret_label{alg_cnt} = label;
            ret_basis{alg_cnt} = sol.W;
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
            ret_label{alg_cnt} = label;
            ret_basis{alg_cnt} = sol.W;
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
            label = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label);
            AC = length(find(gnd == label))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label;
            ret_basis{alg_cnt} = Vemonmf';
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
            label = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label);
            AC = length(find(gnd == label))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label;
            ret_basis{alg_cnt} = Vonpmf';
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
            ret_label{alg_cnt} = label;
            ret_basis{alg_cnt} = A*Bc;
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
            opts.rho = 1e-4;
            opts.rhomax = 1e10;
            opts.eta = 0.995;
            opts.gamma = 1.1;
            opts.tau = 0.1;


            [Zc, Out] = NS_ONMF(A, opts);
            [~, label] = max(abs(Out.C), [], 2);

            purity = calc_purity(gnd, label);
            nmi = calc_nmi(gnd, label);
            label = best_map(gnd,label);
            MIhat = MutualInfo(gnd,label);
            AC = length(find(gnd == label))/length(gnd);
            % Record
            alg_cpu{alg_cnt} = [alg_cpu{alg_cnt}; toc(cpu0)];
            alg_purity{alg_cnt} = [alg_purity{alg_cnt}; purity];
            alg_nmi{alg_cnt} = [alg_nmi{alg_cnt}; nmi];
            alg_MIhat{alg_cnt} = [alg_MIhat{alg_cnt};  MIhat];
            alg_AC{alg_cnt} = [alg_AC{alg_cnt}; AC];
            ret_label{alg_cnt} = label;
            ret_basis{alg_cnt} = Out.B;
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


    % Set flag for visual plot
    flag_visual_plot = 1;
    if flag_visual_plot
        figure('units','normalized','outerposition',[0 0 1 1]);

        outer_layout = tiledlayout(2, 4, 'TileSpacing', 'Compact', 'Padding', 'Compact', 'TileIndexing', 'columnmajor');

        % Define the layout of the inner subplots within each outer subplot
        inner_rows = 10;
        inner_cols = 7;

        for outer_idx = 1:6

            ax = nexttile(outer_layout);
            axis off;
            
            pos = ax.Position;
            
            width = pos(3) / inner_cols;
            height = pos(4) / inner_rows;
            
            for row = 1:inner_rows
                for col = 1:inner_cols
                    i = col + (row-1)*inner_cols;
                    if (i<=nClass)
                        % Calculate the normalized position for each small "subplot"
                        pos_x = pos(1) + (col - 1) * width;
                        pos_y = pos(2) + (inner_rows - row) * height; % Y axis is flipped for plots
                        
                        % Create axes for each small "subplot" within the current tile
                        inner_ax = axes('Position', [pos_x, pos_y, width, height]);
                        img = reshape(ret_basis{outer_idx}(:, i), [32,32]);
                        imagesc(img)
                        axis(inner_ax, 'off');
                        colormap('gray');
                    end
                end
            end
            
            title_position = [0.5, -0.01, 0];  
            title(ax, alg_name{outer_idx}, 'Position', title_position, 'Units', 'normalized', 'VerticalAlignment', 'top');
        end

        ax = nexttile(outer_layout);
        axis off;

        pos = ax.Position;

        for row = 1:inner_rows
            for col = 1:inner_cols
                i = col + (row-1)*inner_cols;
                if(i <=nClass)

                    pos_x = pos(1) + (col - 1) * width;
                    pos_y = pos(2)/2 + (inner_rows - row) * height; % Y axis is flipped for plots
                    
                    % Create axes for each small "subplot" within the current tile
                    inner_ax = axes('Position', [pos_x, pos_y, width, height]);

                    img = reshape(ret_basis{outer_idx+1}(:, i), [32,32]);
                    imagesc(img)
                    axis(inner_ax, 'off');
                    colormap('gray');
                end
            end
        end

        title_position = [0.5, -0.66, 0];  
        title(ax, alg_name{outer_idx+1}, 'Position', title_position, 'Units', 'normalized', 'VerticalAlignment', 'top');% Create a new figure

        flag_save_plot = 1;
        if flag_save_plot
            data_name = split(data_set{t}, '.');
            save_name = sprintf('output/ret-test-%s.png', data_name{1});
            exportgraphics(gcf, save_name)
        end
        
    end


   

end
