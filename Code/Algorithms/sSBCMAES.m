classdef sSBCMAES < Algorithm
% <Many-task> <Single-objective> <None/Constrained>

%------------------------------- Reference --------------------------------
% @Article{Liaw2019SBO,
%   title      = {Evolutionary Manytasking Optimization Based on Symbiosis in Biocoenosis},
%   author     = {Liaw, Rung-Tzuo and Ting, Chuan-Kang},
%   journal    = {Proceedings of the AAAI Conference on Artificial Intelligence},
%   year       = {2019},
%   month      = {Jul.},
%   number     = {01},
%   pages      = {4295-4303},
%   volume     = {33},
%   doi        = {10.1609/aaai.v33i01.33014295},
% }
%--------------------------------------------------------------------------

properties (SetAccess = private)
    Benefit = 0.25
    Harm = 0.5
    sigma0 = 0.3
end

methods
    function Parameter = getParameter(Algo)
        Parameter = {'Benefit: Beneficial factor', num2str(Algo.Benefit), ...
                'Harm: Harmful factor', num2str(Algo.Harm), ...
                'sigma0', num2str(Algo.sigma0)};
    end

    function Algo = setParameter(Algo, Parameter)
        i = 1;
        Algo.Benefit = str2double(Parameter{i}); i = i + 1;
        Algo.Harm = str2double(Parameter{i}); i = i + 1;
        Algo.sigma0 = str2double(Parameter{i}); i = i + 1;
    end

    function run(Algo, Prob)
        D = max(Prob.D);
        lambda = Prob.N;
        mu = round(lambda / 2); % effective solutions number
        weights = log(mu + 0.5) - log(1:mu);
        weights = weights ./ sum(weights); % weights
        mueff = 1 / sum(weights.^2); % variance effective selection mass
        chiN = sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D^2));
        hth = (1.4 + 2 / (D + 1)) * chiN;
        for t = 1:Prob.T
            % Step size control parameters
            cs{t} = (mueff + 2) / (D + mueff + 3);
            damps{t} = 1 + cs{t} + 2 * max(sqrt((mueff - 1) / (D + 1)) - 1, 0);
            % Covariance update parameters
            cc{t} = 4 / (4 + D);
            ccov{t} = (1 / mueff) * (2 / (D + sqrt(2))^2) + (1 - 1 / mueff) * min(1, (2 * mueff - 1) / ((D + 2)^2 + mueff));
            ccov{t} = (D + 2) / 3 * ccov{t};
            % Initialization
            mDec{t} = mean(unifrnd(zeros(lambda, D), ones(lambda, D)));
            ps{t} = zeros(D, 1);
            pc{t} = zeros(D, 1);
            C{t} = ones(D, 1);
            sigma{t} = Algo.sigma0;
            eigenFE{t} = 0;
            for i = 1:lambda
                population{t}(i) = IndividualSBO();
            end
        end

        % Initialization
        for t = 1:Prob.T
            for i = 1:lambda
                population{t}(i).Dec = mDec{t} + sigma{t} * (sqrt(C{t}) .* randn(D, 1))';
            end
            [population{t}, rank] = Algo.EvaluationAndSort(population{t}, Prob, t);
            for i = 1:length(population{t})
                population{t}(rank(i)).rankO = i;
                population{t}(rank(i)).MFFactor = t;
                population{t}(rank(i)).BelongT = t;
            end
        end

        RIJ = 0.5 * ones(Prob.T, Prob.T); % transfer rates
        MIJ = ones(Prob.T, Prob.T); % Benefit and Benefit
        NIJ = ones(Prob.T, Prob.T); % neutral and neutral
        CIJ = ones(Prob.T, Prob.T); % Harm and Harm
        OIJ = ones(Prob.T, Prob.T); % neutral and Benefit
        PIJ = ones(Prob.T, Prob.T); % Benefit and Harm
        AIJ = ones(Prob.T, Prob.T); % Harm and neutral

        while Algo.notTerminated(Prob, population)
            offspring = population;
            for t = 1:Prob.T
                % generation
                for i = 1:length(population{t})
                    offspring{t}(i).Dec = mDec{t} + sigma{t} * (sqrt(C{t}) .* randn(D, 1))';
                    offspring{t}(i).MFFactor = t;
                    offspring{t}(i).BelongT = t;
                    offspring{t}(i).rankO = population{t}(i).rankO;
                end
            end

            for t = 1:Prob.T
                % knowledge transfer
                [~, transfer_task] = max(RIJ(t, [1:t - 1, t + 1:end])); % find transferred task
                if transfer_task >= t
                    transfer_task = transfer_task + 1;
                end
                if rand() < RIJ(t, transfer_task)
                    Si = floor(lambda * RIJ(t, transfer_task)); % transfer quantity
                    ind1 = randperm(lambda, Si);
                    ind2 = randperm(lambda, Si);
                    for i = 1:Si
                        offspring{t}(ind1(i)).Dec = offspring{transfer_task}(ind2(i)).Dec;
                        offspring{t}(ind1(i)).BelongT = transfer_task;
                    end
                end

                % Evaluation
                [offspring{t}, rank] = Algo.EvaluationAndSort(offspring{t}, Prob, t);
                for i = 1:length(rank)
                    offspring{t}(rank(i)).rankC = i;
                end

                % Update CMA variables
                % Update mean decision variables
                oldDec = mDec{t};
                mDec{t} = weights * offspring{t}(rank(1:mu)).Decs;
                % Update evolution paths
                ps{t} = (1 - cs{t}) * ps{t} + sqrt(cs{t} * (2 - cs{t}) * mueff) * (mDec{t} - oldDec)' ./ sqrt(C{t}) / sigma{t};
                hsig = norm(ps{t}) / sqrt(1 - (1 - cs{t})^(2 * (ceil((Algo.FE - lambda * (t - 1)) / (lambda * Prob.T)) + 1))) < hth;
                pc{t} = (1 - cc{t}) * pc{t} + hsig * sqrt(cc{t} * (2 - cc{t}) * mueff) * (mDec{t} - oldDec)' / sigma{t};
                % Update covariance matrix
                artmp = (offspring{t}(rank(1:mu)).Decs - repmat(oldDec, mu, 1))' / sigma{t};
                delta = (1 - hsig) * cc{t} * (2 - cc{t});
                C{t} = (1 - ccov{t}) * C{t} + (ccov{t} / mueff) * (pc{t}.^2 + delta * C{t}) + ccov{t} * (1 - 1 / mueff) * artmp.^2 * weights';
                % Update step size
                sigma{t} = sigma{t} * exp(cs{t} / damps{t} * (norm(ps{t}) / chiN - 1));

                % selection
                population{t} = [population{t}, offspring{t}];
                [~, rank] = sortrows([population{t}.CVs, population{t}.Objs], [1, 2]);
                population{t} = population{t}(rank(1:lambda));
                rank = 1:lambda;
                for i = 1:length(rank)
                    population{t}(rank(i)).rankO = i;
                end
            end

            for t = 1:Prob.T
                % update symbiosis
                idx = find([offspring{t}.BelongT] ~= t);
                rankC = [offspring{t}(idx).rankC];
                rankO = [offspring{t}(idx).rankO];
                for k = 1:length(idx)
                    if rankC(k) < lambda * Algo.Benefit
                        if rankO(k) < lambda * Algo.Benefit
                            MIJ(t, offspring{t}(idx(k)).BelongT) = MIJ(t, offspring{t}(idx(k)).BelongT) + 1;
                        elseif rankO(k) > lambda * (1 - Algo.Harm)
                            PIJ(t, offspring{t}(idx(k)).BelongT) = PIJ(t, offspring{t}(idx(k)).BelongT) + 1;
                        else
                            OIJ(t, offspring{t}(idx(k)).BelongT) = OIJ(t, offspring{t}(idx(k)).BelongT) + 1;
                        end
                    elseif rankC(k) > lambda * (1 - Algo.Harm)
                        if rankO(k) > lambda * (1 - Algo.Harm)
                            CIJ(t, offspring{t}(idx(k)).BelongT) = CIJ(t, offspring{t}(idx(k)).BelongT) + 1;
                        end
                    else
                        if rankO(k) > lambda * (1 - Algo.Harm)
                            AIJ(t, offspring{t}(idx(k)).BelongT) = AIJ(t, offspring{t}(idx(k)).BelongT) + 1;
                        elseif rankO(k) >= lambda * Algo.Benefit && rankO(k) <= lambda * (1 - Algo.Harm)
                            NIJ(t, offspring{t}(idx(k)).BelongT) = NIJ(t, offspring{t}(idx(k)).BelongT) + 1;
                        end
                    end
                end
            end
            % update transfer rates
            RIJ = (MIJ + OIJ + PIJ) ./ (MIJ + OIJ + PIJ + AIJ + CIJ + NIJ);

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
                tempDec = max(0, min(1, tempDec));
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
