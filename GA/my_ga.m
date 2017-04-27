function [best_fitness, elite, generation, last_generation] = my_ga( ...
    number_of_variables, ...    % 求解问题的参数个数
    fitness_function, ...       % 自定义适应度函数名
    population_size, ...        % 种群规模（每一代个体数目）
    parent_number, ...          % 每一代中保持不变的数目（除了变异）
    mutation_rate, ...          % 变异概率
    maximal_generation, ...     % 最大演化代数
    minimal_cost ...            % 最小目标值（函数值越小，则适应度越高）
)

% 累加概率
% 假设 parent_number = 10
% 分子 parent_number:-1:1 用于生成一个数列
% 分母 sum(parent_number:-1:1) 是一个求和结果（一个数）
%
% 分子 10     9     8     7     6     5     4     3     2     1
% 分母 55
% 相除 0.1818    0.1636    0.1455    0.1273    0.1091    0.0909    0.0727    0.0545    0.0364    0.0182
% 累加 0.1818    0.3455    0.4909    0.6182    0.7273    0.8182    0.8909    0.9455    0.9818    1.0000
%
% 运算结果可以看出
% 累加概率函数是一个从0到1增长得越来越慢的函数
% 因为后面加的概率越来越小（数列是降虚排列的）
cumulative_probabilities = cumsum((parent_number:-1:1) / sum(parent_number:-1:1)); % 1个长度为parent_number的数列

% 最佳适应度
% 每一代的最佳适应度都先初始化为1
best_fitness = ones(maximal_generation, 1);

% 精英
% 每一代的精英的参数值都先初始化为0
elite = zeros(maximal_generation, number_of_variables);

% 子女数量
% 种群数量 - 父母数量（父母即每一代中不发生改变的个体）
child_number = population_size - parent_number; % 每一代子女的数目

% 初始化种群
% population_size 对应矩阵的行，每一行表示1个个体，行数=个体数（种群数量）
% number_of_variables 对应矩阵的列，列数=参数个数（个体特征由这些参数表示）
population = rand(population_size, number_of_variables);

last_generation = 0; % 记录跳出循环时的代数


% 后面的代码都在for循环中
for generation = 1 : maximal_generation % 演化循环开始
    
    % feval把数据带入到一个定义好的函数句柄中计算
    % 把population矩阵带入fitness_function函数计算
    cost = feval(fitness_function, population); % 计算所有个体的适应度（population_size*1的矩阵）

    % index记录排序后每个值原来的行数
    [cost, index] = sort(cost); % 将适应度函数值从小到大排序

    % index(1:parent_number) 
    % 前parent_number个cost较小的个体在种群population中的行数
    % 选出这部分(parent_number个)个体作为父母，其实parent_number对应交叉概率
    population = population(index(1:parent_number), :); % 先保留一部分较优的个体
    % 可以看出population矩阵是不断变化的

    % cost在经过前面的sort排序后，矩阵已经改变为升序的
    % cost(1)即为本代的最佳适应度
    best_fitness(generation) = cost(1); % 记录本代的最佳适应度

    % population矩阵第一行为本代的精英个体
    elite(generation, :) = population(1, :); % 记录本代的最优解（精英）

    % 若本代的最优解已足够好，则停止演化
    if best_fitness(generation) < minimal_cost; 
        last_generation = generation;
        break; 
    end
    
    % 交叉变异产生新的种群

    % 染色体交叉开始
    for child = 1:2:child_number % 步长为2是因为每一次交叉会产生2个孩子
        
        % cumulative_probabilities长度为parent_number
        % 从中随机选择2个父母出来  (child+parent_number)%parent_number
        mother = find(cumulative_probabilities > rand, 1); % 选择一个较优秀的母亲
        father = find(cumulative_probabilities > rand, 1); % 选择一个较优秀的父亲
        
        % ceil（天花板）向上取整
        % rand 生成一个随机数
        % 即随机选择了一列，这一列的值交换
        crossover_point = ceil(rand*number_of_variables); % 随机地确定一个染色体交叉点
        
        % 假如crossover_point=3, number_of_variables=5
        % mask1 = 1     1     1     0     0
        % mask2 = 0     0     0     1     1
        mask1 = [ones(1, crossover_point), zeros(1, number_of_variables - crossover_point)];
        mask2 = not(mask1);
        
        % 获取分开的4段染色体
        % 注意是 .*
        mother_1 = mask1 .* population(mother, :); % 母亲染色体的前部分
        mother_2 = mask2 .* population(mother, :); % 母亲染色体的后部分
        
        father_1 = mask1 .* population(father, :); % 父亲染色体的前部分
        father_2 = mask2 .* population(father, :); % 父亲染色体的后部分
        
        % 得到下一代
        population(parent_number + child, :) = mother_1 + father_2; % 一个孩子
        population(parent_number+child+1, :) = mother_2 + father_1; % 另一个孩子
        
    end % 染色体交叉结束
    
    
    % 染色体变异开始
    
    % 变异种群
    mutation_population = population(2:population_size, :); % 精英不参与变异，所以从2开始
    
    number_of_elements = (population_size - 1) * number_of_variables; % 全部基因数目
    number_of_mutations = ceil(number_of_elements * mutation_rate); % 变异的基因数目（基因总数*变异率）
    
    % rand(1, number_of_mutations) 生成number_of_mutations个随机数(范围0-1)组成的矩阵(1*number_of_mutations)
    % 数乘后，矩阵每个元素表示发生改变的基因的位置（元素在矩阵中的一维坐标）
    mutation_points = ceil(number_of_elements * rand(1, number_of_mutations)); % 确定要变异的基因
    
    % 被选中的基因都被一个随机数替代，完成变异
    mutation_population(mutation_points) = rand(1, number_of_mutations); % 对选中的基因进行变异操作
    
    population(2:population_size, :) = mutation_population; % 发生变异之后的种群
    
    % 染色体变异结束
   
end % 演化循环结束

