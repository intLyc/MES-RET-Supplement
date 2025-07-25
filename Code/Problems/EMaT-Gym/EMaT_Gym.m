classdef EMaT_Gym < Problem
% <Many-task> <Single-objective> <None>

% Two-layer neural network with shared hidden layer
% D = Obs * h + h + h * h + h + h * Act + Act
% D = (Obs + Act + h + 2) * h + Act

% |-------------------------------------------------------------------------------------------------------------------------|
% |         | Name                      |    Obs    |  Action  | h =   16 | h =    64 | h =   128 |  Obs Type  |  Act Type  |
% |--------------------- Classic Control -----------------------------------------------------------------------------------|
% | Task  1 | MountainCarContinuous-v0  | Obs =   2 | Act =  1 | D =  337 | D =  4417 | D = 17025 | Continuous | Continuous |
% | Task  2 | MountainCar-v0            | Obs =   2 | Act =  3 | D =  371 | D =  4547 | D = 17283 | Continuous |   Discrete |
% | Task  3 | Pendulum-v1               | Obs =   3 | Act =  1 | D =  353 | D =  4481 | D = 17153 | Continuous | Continuous |
% | Task  4 | CartPole-v1               | Obs =   4 | Act =  2 | D =  386 | D =  4610 | D = 17410 | Continuous |   Discrete |
% | Task  5 | Acrobot-v1                | Obs =   6 | Act =  3 | D =  435 | D =  4803 | D = 17795 | Continuous |   Discrete |
% |------------------------------ Box2D ------------------------------------------------------------------------------------|
% | Task  6 | LunarLander-v3            | Obs =   8 | Act =  4 | D =  484 | D =  4996 | D = 18180 | Continuous |   Discrete |
% | Task  7 | BipedalWalker-v3          | Obs =  24 | Act =  4 | D =  740 | D =  6020 | D = 20228 | Continuous | Continuous |
% |----------------------------- Mujoco ------------------------------------------------------------------------------------|
% | Task  8 | InvertedPendulum-v5       | Obs =   4 | Act =  1 | D =  369 | D =  4545 | D = 17281 | Continuous | Continuous |
% | Task  9 | InvertedDoublePendulum-v5 | Obs =   9 | Act =  1 | D =  449 | D =  4865 | D = 17921 | Continuous | Continuous |
% | Task 10 | Reacher-v5                | Obs =  10 | Act =  2 | D =  482 | D =  4994 | D = 18178 | Continuous | Continuous |
% | Task 11 | Pusher-v5                 | Obs =  23 | Act =  7 | D =  775 | D = 11464 | D = 20487 | Continuous | Continuous |
% | Task 12 | HalfCheetah-v5            | Obs =  17 | Act =  6 | D =  662 | D =  6151 | D = 19590 | Continuous | Continuous |
% | Task 13 | Hopper-v5                 | Obs =  11 | Act =  3 | D =  515 | D =  5123 | D = 18435 | Continuous | Continuous |
% | Task 14 | Walker2d-v5               | Obs =  17 | Act =  6 | D =  662 | D =  5702 | D = 19590 | Continuous | Continuous |
% | Task 15 | Swimmer-v5                | Obs =   8 | Act =  2 | D =  450 | D =  4866 | D = 17922 | Continuous | Continuous |
% | Task 16 | Ant-v5                    | Obs = 105 | Act =  8 | D = 2104 | D = 11464 | D = 31112 | Continuous | Continuous |
% | Task 17 | Humanoid-v5               | Obs = 348 | Act = 17 | D = [No] | D =  [No] | D =  [No] | Continuous | Continuous |
% | Task 18 | HumanoidStandup-v5        | Obs = 348 | Act = 17 | D = [No] | D =  [No] | D =  [No] | Continuous | Continuous |
% |-------------------------------------------------------------------------------------------------------------------------|

properties (Access = public)
    hiddenSize = 16
    numRollouts = 3
    multiProcess = false
    tasks
end

methods
    function Prob = EMaT_Gym(varargin)
        Prob = Prob@Problem(varargin);
        Prob.maxFE = 18 * 50 * 500;
        Prob.GlobalBest = false;
        Prob.Bounded = false;
    end

    function Parameter = getParameter(Prob)
        Parameter = {'HiddenSize', num2str(Prob.hiddenSize), ...
                'NumRollouts', num2str(Prob.numRollouts), ...
                'MultiProcess', num2str(Prob.multiProcess)};
        Parameter = [Prob.getRunParameter(), Parameter];
    end

    function Prob = setParameter(Prob, Parameter)
        Prob.hiddenSize = str2double(Parameter{3});
        Prob.numRollouts = str2double(Parameter{4});
        Prob.multiProcess = str2double(Parameter{5});
        Prob.setRunParameter(Parameter(1:2));
    end

    function setTasks(Prob)
        if count(py.sys.path, 'Problems/Test/EMaT-Gym') == 0
            py.sys.path().append('Problems/Test/EMaT-Gym');
        end
        py.importlib.import_module('gym_runner');
        py.gym_runner.reset_global_normalizer();

        taskNames = {"MountainCarContinuous-v0", "MountainCar-v0", ...
                "Pendulum-v1", "CartPole-v1", "Acrobot-v1", ...
                "LunarLander-v3", "BipedalWalker-v3", ...
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", ...
                "Reacher-v5", "Pusher-v5", ...
                "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", ...
                "Swimmer-v5", "Ant-v5", ...
                "Humanoid-v5", "HumanoidStandup-v5"};

        Prob.T = length(taskNames);
        gymModule = py.importlib.import_module('gymnasium');
        Prob.tasks = repmat(struct('name', '', 'obsDim', 0, 'actDim', 0, 'paramCount', 0), 1, Prob.T);
        for i = 1:Prob.T
            env = gymModule.make(taskNames{i});
            Prob.tasks(i).name = taskNames{i};

            obsSpace = env.observation_space;
            if isa(obsSpace, 'py.gymnasium.spaces.discrete.Discrete')
                Prob.tasks(i).obsDim = double(obsSpace.n);
            else
                Prob.tasks(i).obsDim = double(obsSpace.shape{1});
            end

            if isa(env.action_space, 'py.gymnasium.spaces.discrete.Discrete')
                Prob.tasks(i).actDim = double(env.action_space.n);
            else
                Prob.tasks(i).actDim = double(env.action_space.shape{1});
            end

            % Parameter count for a simple policy network with discrete actions
            nW1 = Prob.tasks(i).obsDim * Prob.hiddenSize;
            nB1 = Prob.hiddenSize;
            nW2 = Prob.hiddenSize * Prob.hiddenSize;
            nB2 = Prob.hiddenSize;
            nW3 = Prob.hiddenSize * Prob.tasks(i).actDim; % Output layer size is the number of actions
            nB3 = Prob.tasks(i).actDim;
            Prob.tasks(i).paramCount = nW1 + nB1 + nW2 + nB2 + nW3 + nB3;
            env.close();
        end

        for i = 1:Prob.T
            env_name = Prob.tasks(i).name;
            Prob.M(i) = 1;
            Prob.D(i) = Prob.tasks(i).paramCount;
            Prob.Fnc{i} = @(x)TaskWrapper(i, x, Prob.D(i), Prob.hiddenSize, Prob.numRollouts, Prob.multiProcess);
            Prob.Lb{i} = -5 * ones(1, Prob.D(i));
            Prob.Ub{i} = 5 * ones(1, Prob.D(i));
        end
    end
end
end

function [Obj, Con] = TaskWrapper(taskIdx, x, dim, hiddenSize, numRollouts, multiProcess)
if any(isnan(x(:, 1:dim)))
    Obj = inf(size(x, 1), 1);
    Con = zeros(size(x, 1), 1);
    return;
end
params_matrix = py.numpy.array(x(:, 1:dim), dtype = py.numpy.float32);
rewards_py = py.gym_runner.run_episode( ...
    int32(taskIdx - 1), params_matrix, int32(hiddenSize), int32(numRollouts), logical(multiProcess) ...
);
Obj = -double(py.array.array('d', rewards_py))';
Con = zeros(size(x, 1), 1);
end
