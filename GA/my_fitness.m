function y = my_fitness(population)
% population是随机数[0,1]矩阵，下面的操作改变范围为[-1,1]
population = 2 * (population - 0.5); 
y = sum(population.^2, 2); % 行的平方和
