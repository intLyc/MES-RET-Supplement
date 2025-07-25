classdef MES_RET < Algorithm
% <Multi-task/Many-task> <Single-objective> <None/Constrained>

properties (SetAccess = private)
    sigma0 = 0.3
    tau = 1
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
                if CMA.StopFlag(t), continue; end
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
            if Algo.Best{t}.CV <= 0
                old_best = CMA.PreviousPara{t}.fitness(2);
                init_best = CMA.obj_init{t};
                new_best = Algo.Best{t}.Obj;
            else
                old_best = CMA.PreviousPara{t}.fitness(1);
                init_best = CMA.cv_init{t};
                new_best = Algo.Best{t}.CV;
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
            traceC = trace(CMA.C{t});
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
            CMA.sample{t}(i).Dec = CMA.mDec{t} + CMA.sigma{t} * (CMA.B{t} * (CMA.D{t} .* randn(CMA.n{t}, 1)))';
        end

        % Knowledge transfer
        if Algo.tau > 0 && Algo.FE > Prob.N * Prob.T * 2
            if Algo.tau >= 1, tr_num = round(Algo.tau);
            elseif rand() < Algo.tau, tr_num = 1;
            else tr_num = 0;
            end

            %% Phenotypic: Mean decision variable sampling
            m = CMA.mean_mdec(1:CMA.n{t});
            decs = CMA.sample{t}.Decs;
            mStep = mean(sqrt(sum((decs(:, 1:CMA.n{t}) - CMA.mDec{t}).^2, 2)));
            for i = 1:tr_num
                u = m - CMA.mDec{t} + CMA.sigma{t} * (CMA.B{t} * (CMA.D{t} .* randn(CMA.n{t}, 1)))';
                CMA.sample{t}(i).Dec = CMA.mDec{t} + u / norm(u) * mStep;
            end

            %% Genotypic: Distribution variation sampling
            v = CMA.mean_sigma_ratio .* CMA.mean_std_ratio(1:CMA.n{t})
            for i = tr_num + 1:tr_num * 2
                u = v .* CMA.sigma{t} .* sqrt(diag(CMA.C{t}))' .* randn(1, CMA.n{t});
                CMA.sample{t}(i).Dec = CMA.mDec{t} + u;
            end
        end

        [CMA.sample{t}, CMA.rank{t}] = Algo.EvaluationAndSort(CMA.sample{t}, Prob, t, CMA.cv_max{t});
    end

    function CMA = InitCMA(Algo, Prob, lambda)
        % Initialize CMA-ES parameters for T tasks
        for t = 1:Prob.T
            CMA.n{t} = Prob.D(t); % dimension
            CMA.lambda{t} = lambda; % sample points number
            CMA.mu{t} = round(CMA.lambda{t} / 2); % effective solutions number
            CMA.weights{t} = log(CMA.mu{t} + 0.5) - log(1:CMA.mu{t});
            CMA.weights{t} = CMA.weights{t} ./ sum(CMA.weights{t}); % weights
            CMA.mueff = 1 / sum(CMA.weights{t}.^2); % variance effective selection mass
            % expectation of the Euclidean norm of a N(0,I) distributed random vector
            CMA.chiN{t} = sqrt(CMA.n{t}) * (1 - 1 / (4 * CMA.n{t}) + 1 / (21 * CMA.n{t}^2));
            CMA.hth{t} = (1.4 + 2 / (CMA.n{t} + 1)) * CMA.chiN{t};
            % Step size control parameters
            CMA.cs{t} = (CMA.mueff + 2) / (CMA.n{t} + CMA.mueff + 5);
            CMA.damps{t} = 1 + CMA.cs{t} + 2 * max(sqrt((CMA.mueff - 1) / (CMA.n{t} + 1)) - 1, 0);
            % Covariance update parameters
            CMA.cc{t} = (4 + CMA.mueff / CMA.n{t}) / (4 + CMA.n{t} + 2 * CMA.mueff / CMA.n{t});
            CMA.c1{t} = 2 / ((CMA.n{t} + 1.3)^2 + CMA.mueff);
            CMA.cmu{t} = min(1 - CMA.c1{t}, 2 * (CMA.mueff - 2 + 1 / CMA.mueff) / ((CMA.n{t} + 2)^2 + 2 * CMA.mueff / 2));
            % Initialization
            CMA.mDec{t} = mean(unifrnd(zeros(CMA.lambda{t}, CMA.n{t}), ones(CMA.lambda{t}, CMA.n{t})));
            CMA.ps{t} = zeros(CMA.n{t}, 1);
            CMA.pc{t} = zeros(CMA.n{t}, 1);
            CMA.B{t} = eye(CMA.n{t}, CMA.n{t});
            CMA.D{t} = ones(CMA.n{t}, 1);
            CMA.C{t} = CMA.B{t} * diag(CMA.D{t}.^2) * CMA.B{t}';
            CMA.invsqrtC{t} = CMA.B{t} * diag(CMA.D{t}.^-1) * CMA.B{t}';
            CMA.sigma{t} = Algo.sigma0;
            CMA.eigenFE{t} = 0;
            for i = 1:CMA.lambda{t}
                CMA.sample{t}(i) = Individual();
            end
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
            CMA = Algo.SamplingAndTransfer(Prob, CMA, t);
            CMA = Algo.ParameterUpdate(Prob, CMA, t);
            temp_cv = CMA.sample{t}.CVs;
            [~, idx] = sort(temp_cv);
            CMA.cv_max{t} = temp_cv(idx(round(0.2 * length(temp_cv))));
            CMA.obj_init{t} = CMA.sample{t}(CMA.rank{t}(1)).Obj;
            CMA.cv_init{t} = CMA.sample{t}(CMA.rank{t}(1)).CV;
        end
    end

    function CMA = ParameterUpdate(Algo, Prob, CMA, t)
        % Update CMA parameters
        % Update mean decision variables
        oldDec = CMA.mDec{t};
        rankDecs = CMA.sample{t}(CMA.rank{t}(1:CMA.mu{t})).Decs;
        rankDecs = rankDecs(:, 1:CMA.n{t});
        CMA.mDec{t} = CMA.weights{t} * rankDecs;
        % Update evolution paths
        CMA.ps{t} = (1 - CMA.cs{t}) * CMA.ps{t} + sqrt(CMA.cs{t} * (2 - CMA.cs{t}) * CMA.mueff) * CMA.invsqrtC{t} * (CMA.mDec{t} - oldDec)' / CMA.sigma{t};
        hsig = norm(CMA.ps{t}) / sqrt(1 - (1 - CMA.cs{t})^(2 * (ceil((Algo.FE - CMA.lambda{t} * (t - 1)) / (CMA.lambda{t} * Prob.T)) + 1))) < CMA.hth{t};
        CMA.pc{t} = (1 - CMA.cc{t}) * CMA.pc{t} + hsig * sqrt(CMA.cc{t} * (2 - CMA.cc{t}) * CMA.mueff) * (CMA.mDec{t} - oldDec)' / CMA.sigma{t};
        % Update covariance matrix
        artmp = (rankDecs - repmat(oldDec, CMA.mu{t}, 1))' / CMA.sigma{t};
        delta = (1 - hsig) * CMA.cc{t} * (2 - CMA.cc{t});
        CMA.C{t} = (1 - CMA.c1{t} - CMA.cmu{t}) * CMA.C{t} + CMA.c1{t} * (CMA.pc{t} * CMA.pc{t}' + delta * CMA.C{t}) + CMA.cmu{t} * artmp * diag(CMA.weights{t}) * artmp';
        % Update step size
        CMA.sigma{t} = CMA.sigma{t} * exp(CMA.cs{t} / CMA.damps{t} * (norm(CMA.ps{t}) / CMA.chiN{t} - 1));
        % Check distribution correctness
        if (Algo.FE - CMA.lambda{t} * (t - 1)) - CMA.eigenFE{t} > (CMA.lambda{t} * Prob.T) / (CMA.c1{t} + CMA.cmu{t}) / CMA.n{t} / 10 % to achieve O(N^2)
            CMA.eigenFE{t} = Algo.FE;
            if ~(all(~isnan(CMA.C{t}), 'all') && all(~isinf(CMA.C{t}), 'all'))
                CMA.StopFlag(t) = true;
            else
                CMA.C{t} = triu(CMA.C{t}) + triu(CMA.C{t}, 1)'; % enforce symmetry
                [CMA.B{t}, CMA.D{t}] = eig(CMA.C{t}); % eigen decomposition, B==normalized eigenvectors
                if min(diag(CMA.D{t})) < 0
                    CMA.StopFlag(t) = true;
                else
                    CMA.D{t} = sqrt(diag(CMA.D{t})); % D contains standard deviations now
                end
            end
            if CMA.StopFlag(t)
                return;
            end
            CMA.invsqrtC{t} = CMA.B{t} * diag(CMA.D{t}.^-1) * CMA.B{t}';
        end

        % Update knowledge extraction parameters
        if ~isempty(CMA.PreviousPara{t})
            CMA.sigma_ratio(t) = CMA.sigma{t} / CMA.PreviousPara{t}.sigma;
            CMA.std_ratio(t, 1:CMA.n{t}) = sqrt(diag(CMA.C{t}))' ./ CMA.PreviousPara{t}.std;
            CMA.mdec_matrix(t, 1:CMA.n{t}) = CMA.mDec{t};
        end
    end

    function Para = SavePara(Algo, CMA, t)
        % Save parameters for the next generation
        Para = struct();
        Para.std = sqrt(diag(CMA.C{t}))';
        Para.sigma = CMA.sigma{t};
        bestObj = Algo.Best{t}.Obj;
        bestCV = Algo.Best{t}.CV;
        Para.fitness = [bestCV; bestObj];
    end

    function CMA = CheckStop(Algo, Prob, CMA)
        % Check stopping criteria (sigma * max(pc, std) < 1e-12)
        for t = 1:Prob.T
            if all(CMA.sigma{t} * (max(abs(CMA.pc{t}), sqrt(diag(CMA.C{t})))) < 1e-12)
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

        % Epsilon constraint handling for constrained problems
        CVs = sample.CVs;
        if Algo.FE < 0.3 * Prob.maxFE && cv_max > 0
            Ep = cv_max * ((1 - Algo.FE / (0.3 * Prob.maxFE))^8);
            CVs(CVs < Ep) = 0;
        end

        if Prob.Bounded
            % Boundary constraint handling
            boundCVs = zeros(length(sample), 1);
            for i = 1:length(sample)
                tempDec = sample(i).Dec;
                tempDec = max(-0.05, min(1.05, tempDec));
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
