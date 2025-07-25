classdef sMES_RET < Algorithm
% <Multi-task/Many-task> <Single-objective> <None/Constrained>

properties (SetAccess = private)
    sigma0 = 0.3
    tau = 2
end

methods
    function Parameter = getParameter(Algo)
        Parameter = {'sigma0', num2str(Algo.sigma0), ...
                'tau', num2str(Algo.tau)};
    end

    function Algo = setParameter(Algo, Parameter)
        Algo.sigma0 = str2double(Parameter{1});
        Algo.tau = str2double(Parameter{2});
    end

    function run(Algo, Prob)
        % Initialization
        CMA = Algo.InitCMA(Prob, Prob.N);

        % Main Loop
        while Algo.notTerminated(Prob, CMA.sample)
            % Self Evolution
            for t = 1:Prob.T
                if CMA.StopFlag(t), continue; end % Skip the task that has stopped
                CMA.PreviousPara{t} = Algo.SavePara(CMA, t);
                CMA = Algo.SamplingAndTransfer(Prob, CMA, t);
                CMA = Algo.ParameterUpdate(Prob, CMA, t);
            end

            % Reward Calculation
            if rand() < 1 - Algo.FE / Prob.maxFE
                Reward = Algo.ComputeRewardFit(Prob, CMA);
            else
                Reward = Algo.ComputeRewardDiv(Prob, CMA);
            end

            % Reward-weighted Knowledge Extraction
            idx = ~CMA.StopFlag;
            if sum(Reward(idx)) < 1e-12
                CMA.mean_sigma_ratio = mean(CMA.sigma_ratio(idx)');
            else
                CMA.mean_sigma_ratio = sum(Reward(idx) ./ sum(Reward(idx)) .* CMA.sigma_ratio(idx)');
            end
            for j = 1:max(Prob.D)
                idx = ~CMA.StopFlag & ~isnan(CMA.std_ratio(:, j))' & ~isnan(CMA.mdec_matrix(:, j))';
                if isempty(find(idx, 1))
                    CMA.mean_std_ratio(j) = NaN;
                    CMA.mean_mdec(j) = NaN;
                elseif sum(Reward(idx)) < 1e-12
                    CMA.mean_std_ratio(j) = mean(CMA.std_ratio(idx, j)');
                    CMA.mean_mdec(j) = mean(CMA.mdec_matrix(idx, j)');
                else
                    CMA.mean_std_ratio(j) = sum(Reward(idx) ./ sum(Reward(idx)) .* CMA.std_ratio(idx, j)');
                    CMA.mean_mdec(j) = sum(Reward(idx) ./ sum(Reward(idx)) .* CMA.mdec_matrix(idx, j)');
                end
            end

            % Reward-weighted Evaluation
            for x = 1:Prob.T
                t = RouletteSelection(Reward);
                if CMA.StopFlag(t), continue; end
                CMA.PreviousPara{t} = Algo.SavePara(CMA, t);
                CMA = Algo.SamplingAndTransfer(Prob, CMA, t);
                CMA = Algo.ParameterUpdate(Prob, CMA, t);
            end

            % Check Stopping Criteria
            CMA = Algo.CheckStop(Prob, CMA);

            % Save the best solution on EMaT-Gym
            if Algo.FE >= Prob.maxFE && contains(Prob.Name, 'EMaT-Gym')
                randNum = randi(100000);

                % save mean decision variables
                Dec = nan(Prob.T, max(Prob.D));
                for t = 1:Prob.T
                    Dec(t, 1:Prob.D(t)) = Algo.Best{t}.Dec(1:Prob.D(t)) .* (Prob.Ub{t} - Prob.Lb{t}) + Prob.Lb{t};
                end
                save([Algo.Name, ' Dec ', num2str(randNum)], 'Dec');

                normalizer = cell(1, Prob.T);
                for t = 1:Prob.T
                    env_name = char(Prob.tasks(t).name);
                    if contains(Prob.Name, 'KAN')
                        py_stats = py.gym_runner_kan.get_normalizer_stats(env_name);
                    else
                        py_stats = py.gym_runner.get_normalizer_stats(env_name);
                    end
                    normalizer{t}.name = env_name;
                    normalizer{t}.mean = double(py.array.array('d', py_stats{'mean'}));
                    normalizer{t}.std = double(py.array.array('d', py_stats{'std'}));
                end
                save([Algo.Name, ' Normalizer ', num2str(randNum)], 'normalizer');
            end
        end
    end

    function RewardFit = ComputeRewardFit(Algo, Prob, CMA)
        % RewardFit: Reward for objective improvement
        % Normalized fitness improvement
        for t = 1:Prob.T
            if CMA.StopFlag(t)
                improveFit(t) = 0;
                continue;
            end
            if CMA.gBest{t}.CV <= 0
                old_best = CMA.PreviousPara{t}.fitness(2);
                init_best = CMA.obj_init{t};
                new_best = CMA.gBest{t}.Obj;
            else
                old_best = CMA.PreviousPara{t}.fitness(1);
                init_best = CMA.cv_init{t};
                new_best = CMA.gBest{t}.CV;
            end
            improveFit(t) = max(0, old_best - new_best) / (init_best - old_best +1e-12);
        end
        sumScore = sum(improveFit);
        if sumScore < 1e-12
            RewardFit = ones(1, Prob.T) / Prob.T;
        else
            RewardFit = improveFit / sumScore;
        end
        if any(isnan(RewardFit))
            RewardFit = ones(1, Prob.T) / Prob.T;
        end
    end

    function RewardDiv = ComputeRewardDiv(Algo, Prob, CMA)
        % RewardDiv: Reward for diversity
        % Normalized trace of covariance matrix
        for t = 1:Prob.T
            if CMA.StopFlag(t)
                diversity(t) = 0;
                continue;
            end
            sigma = CMA.PreviousPara{t}.sigma;
            traceC = sum(CMA.PreviousPara{t}.std.^2);
            diversity(t) = sigma * traceC / CMA.n{t};
            sigma = CMA.sigma{t};
            traceC = sum(CMA.C{t});
            diversity(t) = diversity(t) + sigma * traceC / CMA.n{t};
        end
        diversity = (diversity - min(diversity)) / (max(diversity) - min(diversity) +1e-12);
        sumDiv = sum(diversity);
        if sumDiv < 1e-12
            RewardDiv = ones(1, Prob.T) / Prob.T;
        else
            RewardDiv = diversity / sumDiv;
        end
        if any(isnan(RewardDiv))
            RewardDiv = ones(1, Prob.T) / Prob.T;
        end
    end

    function CMA = SamplingAndTransfer(Algo, Prob, CMA, t)
        % Sample new solutions
        for i = 1:CMA.lambda{t}
            CMA.sample{t}(i).Dec = CMA.mDec{t} + CMA.sigma{t} * (sqrt(CMA.C{t}) .* randn(CMA.n{t}, 1))';
        end

        % Knowledge transfer
        if Algo.tau > 0 && Algo.FE > Prob.N * Prob.T * 2
            if Algo.tau >= 1, tr_num = round(Algo.tau);
            elseif rand() < Algo.tau, tr_num = 1;
            else tr_num = 0;
            end

            %% Phenotypic: Mean decision variable sampling
            m = CMA.mean_mdec(1:CMA.n{t});
            if any(isnan(m)), m = CMA.mDec{t}; end
            decs = CMA.sample{t}.Decs;
            mStep = mean(sqrt(sum((decs(:, 1:CMA.n{t}) - CMA.mDec{t}).^2, 2)));
            for i = 1:tr_num
                u = m - CMA.mDec{t} + CMA.sigma{t} * (sqrt(CMA.C{t}) .* randn(CMA.n{t}, 1))';
                CMA.sample{t}(i).Dec = CMA.mDec{t} + u / norm(u) * mStep;
            end

            %% Genotypic: Distribution variation sampling
            v = CMA.mean_sigma_ratio .* CMA.mean_std_ratio(1:CMA.n{t})
            if any(isnan(v)), v = ones(1, CMA.n{t}); end
            for i = tr_num + 1:tr_num * 2
                u = v .* CMA.sigma{t} .* sqrt(CMA.C{t})' .* randn(1, CMA.n{t});
                CMA.sample{t}(i).Dec = CMA.mDec{t} + u;
            end
        end

        [CMA.sample{t}, CMA.rank{t}] = Algo.EvaluationAndSort(CMA.sample{t}, Prob, t, CMA.cv_max{t});
        if ~isempty(CMA.gBest{t}) && (Algo.Best{t}.CV < CMA.gBest{t}.CV || ...
                (Algo.Best{t}.CV == CMA.gBest{t}.CV && Algo.Best{t}.Obj < CMA.gBest{t}.Obj))
            CMA.gBest{t} = Algo.Best{t};
        end
    end

    function CMA = InitCMA(Algo, Prob, lambda)
        % Initialize CMA-ES parameters for T tasks
        for t = 1:Prob.T
            CMA.n{t} = Prob.D(t); % dimension
            CMA.lambda{t} = lambda; % sample points number
            CMA.mu{t} = round(CMA.lambda{t} / 2); % effective solutions number
            CMA.weights{t} = log(CMA.mu{t} + 0.5) - log(1:CMA.mu{t});
            CMA.weights{t} = CMA.weights{t} ./ sum(CMA.weights{t}); % weights
            CMA.mueff{t} = 1 / sum(CMA.weights{t}.^2); % variance effective selection mass
            % expectation of the Euclidean norm of a N(0,I) distributed random vector
            CMA.chiN{t} = sqrt(CMA.n{t}) * (1 - 1 / (4 * CMA.n{t}) + 1 / (21 * CMA.n{t}^2));
            CMA.hth{t} = (1.4 + 2 / (CMA.n{t} + 1)) * CMA.chiN{t};
            % Step size control parameters
            CMA.cs{t} = (CMA.mueff{t} + 2) / (CMA.n{t} + CMA.mueff{t} + 3);
            CMA.damps{t} = 1 + CMA.cs{t} + 2 * max(sqrt((CMA.mueff{t} - 1) / (CMA.n{t} + 1)) - 1, 0);
            % Covariance update parameters
            CMA.cc{t} = 4 / (4 + CMA.n{t});
            CMA.ccov{t} = (1 / CMA.mueff{t}) * (2 / (CMA.n{t} + sqrt(2))^2) + (1 - 1 / CMA.mueff{t}) * min(1, (2 * CMA.mueff{t} - 1) / ((CMA.n{t} + 2)^2 + CMA.mueff{t}));
            CMA.ccov{t} = (CMA.n{t} + 2) / 3 * CMA.ccov{t};
            % Initialization
            CMA.mDec{t} = mean(unifrnd(zeros(CMA.lambda{t}, CMA.n{t}), ones(CMA.lambda{t}, CMA.n{t})));
            CMA.ps{t} = zeros(CMA.n{t}, 1);
            CMA.pc{t} = zeros(CMA.n{t}, 1);
            CMA.C{t} = ones(CMA.n{t}, 1);
            CMA.sigma{t} = Algo.sigma0;
            for i = 1:CMA.lambda{t}
                CMA.sample{t}(i) = Individual();
            end
            CMA.RecordFit{t} = [];
            CMA.PreviousPara{t} = [];
            CMA.obj_init{t} = 0;
            CMA.cv_init{t} = 0;
            CMA.cv_max{t} = 0;
        end

        % Initialize knowledge extraction parameters
        CMA.sigma_ratio = ones(Prob.T, 1);
        CMA.std_ratio = nan(Prob.T, max(Prob.D));
        CMA.mdec_matrix = nan(Prob.T, max(Prob.D));
        CMA.StopFlag = false(1, Prob.T);

        % Initial sampling
        for t = 1:Prob.T
            CMA.gBest{t} = [];
            CMA = Algo.Sampling(Prob, CMA, t, 0);
            CMA = Algo.ParameterUpdate(Prob, CMA, t);
            temp_cv = CMA.sample{t}.CVs;
            [~, idx] = sort(temp_cv);
            CMA.cv_max{t} = temp_cv(idx(round(0.2 * length(temp_cv))));
            CMA.gBest{t} = Algo.Best{t};
            CMA.obj_init{t} = CMA.gBest{t}.Obj;
            CMA.cv_init{t} = CMA.gBest{t}.CV;
        end
    end

    function CMA = ParameterUpdate(Algo, Prob, CMA, t)
        % Update CMA parameters
        % Update mean decision variables
        oldDec = CMA.mDec{t};
        rankDecs = CMA.sample{t}(CMA.rank{t}(1:CMA.mu{t})).Decs;
        rankDecs = rankDecs(:, 1:CMA.n{t});
        CMA.mDec{t} = CMA.weights{t} * rankDecs;
        % CMA.mDec{t} = max(0, min(1, CMA.mDec{t}));
        % Update evolution paths
        CMA.ps{t} = (1 - CMA.cs{t}) * CMA.ps{t} + sqrt(CMA.cs{t} * (2 - CMA.cs{t}) * CMA.mueff{t}) * (CMA.mDec{t} - oldDec)' ./ sqrt(CMA.C{t}) / CMA.sigma{t};
        hsig = norm(CMA.ps{t}) / sqrt(1 - (1 - CMA.cs{t})^(2 * (ceil((Algo.FE - CMA.lambda{t} * (t - 1)) / (CMA.lambda{t} * Prob.T)) + 1))) < CMA.hth{t};
        CMA.pc{t} = (1 - CMA.cc{t}) * CMA.pc{t} + hsig * sqrt(CMA.cc{t} * (2 - CMA.cc{t}) * CMA.mueff{t}) * (CMA.mDec{t} - oldDec)' / CMA.sigma{t};
        % Update covariance matrix
        artmp = (rankDecs - repmat(oldDec, CMA.mu{t}, 1))' / CMA.sigma{t};
        delta = (1 - hsig) * CMA.cc{t} * (2 - CMA.cc{t});
        CMA.C{t} = (1 - CMA.ccov{t}) * CMA.C{t} + (CMA.ccov{t} / CMA.mueff{t}) * (CMA.pc{t}.^2 + delta * CMA.C{t}) + CMA.ccov{t} * (1 - 1 / CMA.mueff{t}) * artmp.^2 * CMA.weights{t}';
        % Update step size
        CMA.sigma{t} = CMA.sigma{t} * exp(CMA.cs{t} / CMA.damps{t} * (norm(CMA.ps{t}) / CMA.chiN{t} - 1));

        if ~isempty(CMA.PreviousPara{t})
            CMA.sigma_ratio(t) = CMA.sigma{t} / CMA.PreviousPara{t}.sigma;
            CMA.std_ratio(t, 1:CMA.n{t}) = sqrt(CMA.C{t})' ./ CMA.PreviousPara{t}.std;
            CMA.mdec_matrix(t, 1:CMA.n{t}) = CMA.mDec{t};
        end
    end

    function Para = SavePara(Algo, CMA, t)
        % Save parameters for the next generation
        Para = struct();
        Para.std = sqrt(CMA.C{t})';
        Para.sigma = CMA.sigma{t};
        bestObj = CMA.gBest{t}.Obj;
        bestCV = CMA.gBest{t}.CV;
        Para.fitness = [bestCV; bestObj];
    end

    function CMA = CheckStop(Algo, Prob, CMA)
        % Check stopping criteria (sigma * max(pc, std) < 1e-12)
        for t = 1:Prob.T
            if all(CMA.sigma{t} * (max(abs(CMA.pc{t}), sqrt(CMA.C{t}))) < 1e-12)
                CMA.StopFlag(t) = true;
            end
        end
        if all(CMA.StopFlag)
            CMA = Algo.InitCMA(Prob, CMA.lambda{1} * 2);
        end
    end

    function [sample, rank] = EvaluationAndSort(Algo, sample, Prob, t, cv_max)
        % Evaluation
        sample = Algo.Evaluation(sample, Prob, t);

        % Epsilon constraint handling
        CVs = sample.CVs;
        if Algo.FE < 0.3 * Prob.maxFE && cv_max > 0
            Ep = cv_max * ((1 - Algo.FE / (0.3 * Prob.maxFE))^8);
            CVs(CVs < Ep) = 0;
        end

        if Prob.Bounded
            % Boundary constraint handling
            boundCVs = zeros(length(sample), 1);
            for i = 1:length(sample)
                % Boundary constraint violation for constrained problems
                tempDec = sample(i).Dec;
                tempDec = max(-0.05, min(1.05, tempDec));
                % sample(i).Dec = tempDec;
                boundCVs(i) = sum((sample(i).Dec - tempDec).^2);
            end
            boundCVs(boundCVs > 0) = boundCVs(boundCVs > 0) + max(CVs);
            [~, rank] = sortrows([CVs + boundCVs, sample.Objs], [1, 2]);
        else
            [~, rank] = sortrows([CVs, sample.Objs], [1, 2]);
        end
    end
end
end
