classdef sOpenAI_ES < Algorithm
% <Many-task> <Single-objective> <None/Constrained>

%------------------------------- Reference --------------------------------
% @Misc{Salimans2017OpenAI-ES,
%   title         = {Evolution Strategies as a Scalable Alternative to Reinforcement Learning},
%   author        = {Tim Salimans and Jonathan Ho and Xi Chen and Szymon Sidor and Ilya Sutskever},
%   year          = {2017},
%   archiveprefix = {arXiv},
%   eprint        = {1703.03864},
%   primaryclass  = {stat.ML},
% }
%--------------------------------------------------------------------------

properties (SetAccess = private)
    sigma = 1
    lr = 1e-3
    momentum = 0.9
end

methods
    function Parameter = getParameter(Algo)
        Parameter = {'sigma', num2str(Algo.sigma), ...
                'learning rate', num2str(Algo.lr), ...
                'momentum', num2str(Algo.momentum)};
    end

    function Algo = setParameter(Algo, Parameter)
        Algo.sigma = str2double(Parameter{1});
        Algo.lr = str2double(Parameter{2});
        Algo.momentum = str2double(Parameter{3});
    end

    function run(Algo, Prob)
        if mod(Prob.N, 2) ~= 0
            N = Prob.N + 1;
        else
            N = Prob.N;
        end

        for t = 1:Prob.T
            range = mean(Prob.Ub{t} - Prob.Lb{t});
            sigma{t} = Algo.sigma / range;
            x{t} = mean(unifrnd(zeros(Prob.D(t), N), ones(Prob.D(t), N)), 2);
            v{t} = zeros(Prob.D(t), 1);
            sample{t}(1:N) = Individual();
        end

        while Algo.notTerminated(Prob, sample)
            for t = 1:Prob.T
                % ---- Antithetic Sampling ----
                Z_half = randn(Prob.D(t), N / 2);
                Z = [Z_half, -Z_half];
                X = repmat(x{t}, 1, N) + sigma{t} * Z;

                % ---- Decode samples ----
                for i = 1:N
                    sample{t}(i).Dec = X(:, i)';
                end

                % ---- Evaluate fitness ----
                mean_sample = Individual();
                mean_sample.Dec = x{t}'; % mean decision variable
                sample{t} = Algo.Evaluation([sample{t}, mean_sample], Prob, t);
                sample{t} = sample{t}(1:N);

                % ---- Centered rank shaping ----
                fitness = [sample{t}.Objs];
                [~, sortIdx] = sort(fitness);
                ranks = zeros(1, N);
                ranks(sortIdx) = N - 1:-1:0; % Minimizing fitness
                shaped = ranks / (N - 1) - 0.5;

                % ---- Gradient estimation ----
                grad = (Z * shaped') / (N * sigma{t});

                % ---- Momentum update ----
                v{t} = Algo.momentum * v{t} + (1 - Algo.momentum) * grad;
                x{t} = x{t} + Algo.lr * v{t};
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
end
end
