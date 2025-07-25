classdef sMTES_KG < Algorithm
% <Multi-task/Many-task> <Single-objective> <None/Constrained>

%------------------------------- Reference --------------------------------
% @Article{Li2023MTES-KG,
%   title    = {Multitask Evolution Strategy with Knowledge-Guided External Sampling},
%   author   = {Li, Yanchi and Gong, Wenyin and Li, Shuijia},
%   journal  = {IEEE Transactions on Evolutionary Computation},
%   year     = {2023},
%   doi      = {10.1109/TEVC.2023.3330265},
% }
%--------------------------------------------------------------------------

properties (SetAccess = private)
    tau0 = 2
    alpha = 0.5
    adjGap = 50
    sigma0 = 0.3
end

methods
    function Parameter = getParameter(Algo)
        Parameter = {'tau0: External Sample Number', num2str(Algo.tau0), ...
                'alpha: DoS/SaS Probability', num2str(Algo.alpha), ...
                'adjGap: Gap of Adjust tau', num2str(Algo.adjGap), ...
                'sigma0', num2str(Algo.sigma0)};
    end

    function Algo = setParameter(Algo, Parameter)
        Algo.tau0 = str2double(Parameter{1});
        Algo.alpha = str2double(Parameter{2});
        Algo.adjGap = str2double(Parameter{3});
        Algo.sigma0 = str2double(Parameter{4});
    end

    function run(Algo, Prob)
        n = max(Prob.D); % dimension
        lambda = Prob.N; % sample points number
        mu = round(lambda / 2); % effective solutions number
        weights = log(mu + 0.5) - log(1:mu);
        weights = weights ./ sum(weights); % weights
        mueff = 1 / sum(weights.^2); % variance effective selection mass
        % expectation of the Euclidean norm of a N(0,I) distributed random vector
        chiN = sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n^2));
        hth = (1.4 + 2 / (n + 1)) * chiN;
        for t = 1:Prob.T
            % Step size control parameters
            cs{t} = (mueff + 2) / (n + mueff + 3);
            damps{t} = 1 + cs{t} + 2 * max(sqrt((mueff - 1) / (n + 1)) - 1, 0);
            % Covariance update parameters
            cc{t} = 4 / (4 + n);
            ccov{t} = (1 / mueff) * (2 / (n + sqrt(2))^2) + (1 - 1 / mueff) * min(1, (2 * mueff - 1) / ((n + 2)^2 + mueff));
            ccov{t} = (n + 2) / 3 * ccov{t};
            % Initialization
            mDec{t} = mean(unifrnd(zeros(lambda, n), ones(lambda, n)));
            ps{t} = zeros(n, 1);
            pc{t} = zeros(n, 1);
            C{t} = ones(n, 1);
            sigma{t} = Algo.sigma0;
            eigenFE{t} = 0;
            for i = 1:lambda
                sample{t}(i) = Individual();
            end
            mStep{t} = 0; % mean sample step
            numExS{t} = []; % external sapmle number memory
            sucExS{t} = []; % external sapmle success number memory
            tau(t) = Algo.tau0; % external sample number
            record_tau{t} = tau(t);
        end
        rank = {};

        while Algo.notTerminated(Prob, sample)
            %% Sample new solutions
            oldsample = sample;
            sample{t} = sample{t}(1:lambda);
            for t = 1:Prob.T
                for i = 1:lambda
                    sample{t}(i).Dec = mDec{t} + sigma{t} * (sqrt(C{t}) .* randn(n, 1))';
                end
                mStep{t} = mean(sqrt(sum((sample{t}.Decs - mDec{t}).^2, 2)));
            end

            %% Sample external solutions
            for t = 1:Prob.T
                % Select auxiliary task
                idx = 1:Prob.T; idx(t) = [];
                k = idx(randi(end));

                for i = lambda + 1:lambda + tau(t)
                    if Algo.Gen < 2
                        sample{t}(i).Dec = mDec{t} + sigma{t} * (sqrt(C{t}) .* randn(n, 1))';
                        continue;
                    end
                    if rand() < Algo.alpha
                        % Optimal domain knowledge-guided external sampling (DoS)
                        sample_k = mDec{k} + sigma{k} * (sqrt(C{k}) .* randn(n, 1))';
                        vec = (sample_k - mDec{t});
                        if norm(vec) < mStep{t}
                            sample{t}(i).Dec = sample_k;
                        else
                            uni_vec = vec ./ norm(vec);
                            sample{t}(i).Dec = mDec{t} + uni_vec * mStep{t};
                        end
                    else
                        % Function shape knowledge-guided external sampling (SaS)
                        idx = 1:mu; idx(randi(end)) = [];
                        vec = mean(oldsample{k}(rank{k}(idx)).Decs);
                        vec = (vec - mDec{k}) ./ sigma{k};
                        sample{t}(i).Dec = mDec{t} + sigma{t} * (sqrt(C{t}) .* (sqrt(C{k}).^-1 .* vec'))';
                    end
                end
            end

            %% Update algorithm parameters
            for t = 1:Prob.T
                [sample{t}, rank{t}] = Algo.EvaluationAndSort(sample{t}, Prob, t);

                % Storage number and success of external samples
                numExS{t}(Algo.Gen) = tau(t);
                sucExS{t}(Algo.Gen) = length(find(rank{t}(1:mu) > lambda));

                % Negative transfer mitigation
                if mod(Algo.Gen, Algo.adjGap) == 0
                    numAll = sum(numExS{t}(Algo.Gen - Algo.adjGap + 1:Algo.Gen));
                    sucAll = sum(sucExS{t}(Algo.Gen - Algo.adjGap + 1:Algo.Gen));

                    if (numAll > 0 && sucAll / numAll > 0.5) || (numAll == 0)
                        tau(t) = min([Algo.tau0, tau(t) + 1]);
                    else
                        tau(t) = max(0, tau(t) - 1);
                    end
                end

                % Update CMA parameters
                % Update mean decision variables
                oldDec = mDec{t};
                mDec{t} = weights * sample{t}(rank{t}(1:mu)).Decs;
                % Update evolution paths
                ps{t} = (1 - cs{t}) * ps{t} + sqrt(cs{t} * (2 - cs{t}) * mueff) * (mDec{t} - oldDec)' ./ sqrt(C{t}) / sigma{t};
                hsig = norm(ps{t}) / sqrt(1 - (1 - cs{t})^(2 * (ceil((Algo.FE - lambda * (t - 1)) / (lambda * Prob.T)) + 1))) < hth;
                pc{t} = (1 - cc{t}) * pc{t} + hsig * sqrt(cc{t} * (2 - cc{t}) * mueff) * (mDec{t} - oldDec)' / sigma{t};
                % Update covariance matrix
                artmp = (sample{t}(rank{t}(1:mu)).Decs - repmat(oldDec, mu, 1))' / sigma{t};
                delta = (1 - hsig) * cc{t} * (2 - cc{t});
                C{t} = (1 - ccov{t}) * C{t} + (ccov{t} / mueff) * (pc{t}.^2 + delta * C{t}) + ccov{t} * (1 - 1 / mueff) * artmp.^2 * weights';
                % Update step size
                sigma{t} = sigma{t} * exp(cs{t} / damps{t} * (norm(ps{t}) / chiN - 1));

                record_tau{t} = [record_tau{t}; tau(t)];
            end

            if contains(Prob.Name, 'EMaT-Gym') && Algo.FE >= Prob.maxFE
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
                    py_stats = py.gym_runner.get_normalizer_stats(env_name);
                    normalizer{t}.name = env_name;
                    normalizer{t}.mean = double(py.array.array('d', py_stats{'mean'}));
                    normalizer{t}.std = double(py.array.array('d', py_stats{'std'}));
                end
                save([Algo.Name, ' Normalizer ', num2str(randNum)], 'normalizer');
            end
        end
    end

    function [sample, rank] = EvaluationAndSort(Algo, sample, Prob, t)
        sample = Algo.Evaluation(sample, Prob, t);
        if Prob.Bounded
            % Boundary constraint handling
            boundCVs = zeros(length(sample), 1);
            for i = 1:length(sample)
                tempDec = sample(i).Dec;
                tempDec = max(-0.05, min(1.05, tempDec));
                boundCVs(i) = sum((sample(i).Dec - tempDec).^2);
            end
            boundCVs(boundCVs > 0) = boundCVs(boundCVs > 0) + max(sample.CVs);
            [~, rank] = sortrows([sample.CVs + boundCVs, sample.Objs], [1, 2]);
        else
            [~, rank] = sortrows([sample.CVs, sample.Objs], [1, 2]);
        end
    end
end
end
