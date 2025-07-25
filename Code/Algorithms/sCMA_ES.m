classdef sCMA_ES < Algorithm
% <Many-task> <Single-objective> <None/Constrained>

%------------------------------- Reference --------------------------------
% @InProceedings{Ros2008sep-CMA-ES,
%   title      = {A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity},
%   author     = {Ros, Raymond and Hansen, Nikolaus},
%   booktitle  = {Parallel Problem Solving from Nature -- PPSN X},
%   year       = {2008},
%   address    = {Berlin, Heidelberg},
%   editor     = {Rudolph, G{\"u}nter and Jansen, Thomas and Beume, Nicola and Lucas, Simon and Poloni, Carlo},
%   pages      = {296--305},
%   publisher  = {Springer Berlin Heidelberg},
%   isbn       = {978-3-540-87700-4},
% }
%--------------------------------------------------------------------------

properties (SetAccess = private)
    sigma0 = 0.3
end

methods
    function Parameter = getParameter(Algo)
        Parameter = {'sigma0', num2str(Algo.sigma0)};
    end

    function Algo = setParameter(Algo, Parameter)
        Algo.sigma0 = str2double(Parameter{1});
    end

    function run(Algo, Prob)
        lambda = Prob.N; % sample points number
        mu = round(lambda / 2); % effective solutions number
        weights = log(mu + 0.5) - log(1:mu);
        weights = weights ./ sum(weights); % weights
        mueff = 1 / sum(weights.^2); % variance effective selection mass
        for t = 1:Prob.T
            n{t} = Prob.D(t); % dimension
            % Step size control parameters
            cs{t} = (mueff + 2) / (n{t} + mueff + 3);
            damps{t} = 1 + cs{t} + 2 * max(sqrt((mueff - 1) / (n{t} + 1)) - 1, 0);
            % Covariance update parameters
            cc{t} = 4 / (4 + n{t});
            ccov{t} = (1 / mueff) * (2 / (n{t} + sqrt(2))^2) + (1 - 1 / mueff) * min(1, (2 * mueff - 1) / ((n{t} + 2)^2 + mueff));
            ccov{t} = (n{t} + 2) / 3 * ccov{t};
            % Initialization
            mDec{t} = mean(unifrnd(zeros(lambda, n{t}), ones(lambda, n{t})));
            ps{t} = zeros(n{t}, 1);
            pc{t} = zeros(n{t}, 1);
            C{t} = ones(n{t}, 1);
            sigma{t} = Algo.sigma0;
            chiN{t} = sqrt(n{t}) * (1 - 1 / (4 * n{t}) + 1 / (21 * n{t}^2));
            hth{t} = (1.4 + 2 / (n{t} + 1)) * chiN{t};
            for i = 1:lambda
                sample{t}(i) = Individual();
            end
        end

        while Algo.notTerminated(Prob, sample)
            for t = 1:Prob.T
                % Sample solutions
                for i = 1:lambda
                    sample{t}(i).Dec = mDec{t} + sigma{t} * (sqrt(C{t}) .* randn(n{t}, 1))';
                end
                [sample{t}, rank] = Algo.EvaluationAndSort(sample{t}, Prob, t);

                % Update mean decision variables
                oldDec = mDec{t};
                mDec{t} = weights * sample{t}(rank(1:mu)).Decs;
                % Update evolution paths
                ps{t} = (1 - cs{t}) * ps{t} + sqrt(cs{t} * (2 - cs{t}) * mueff) * (mDec{t} - oldDec)' ./ sqrt(C{t}) / sigma{t};
                hsig = norm(ps{t}) / sqrt(1 - (1 - cs{t})^(2 * (ceil((Algo.FE - lambda * (t - 1)) / (lambda * Prob.T)) + 1))) < hth{t};
                pc{t} = (1 - cc{t}) * pc{t} + hsig * sqrt(cc{t} * (2 - cc{t}) * mueff) * (mDec{t} - oldDec)' / sigma{t};
                % Update covariance matrix
                artmp = (sample{t}(rank(1:mu)).Decs - repmat(oldDec, mu, 1))' / sigma{t};
                delta = (1 - hsig) * cc{t} * (2 - cc{t});
                C{t} = (1 - ccov{t}) * C{t} + (ccov{t} / mueff) * (pc{t}.^2 + delta * C{t}) + ccov{t} * (1 - 1 / mueff) * artmp.^2 * weights';
                % Update step size
                sigma{t} = sigma{t} * exp(cs{t} / damps{t} * (norm(ps{t}) / chiN{t} - 1));
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
