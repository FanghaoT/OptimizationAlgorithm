"Particle Swarm Optimization Algorithm"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


class PSO:
    def __init__(
        self,
        weight = 1,                              #Inertia weight
        lr = (0.5,0.5),                          #Learning rates for individual and global
        maxiter = 100,                           #Maximum number of iterations
        popsize = 100,                           #Population size
        dim = 2,                                 #Dimension of the problem
        poprange = (-5,5),                  #Range of the population
        speedrange = (-0.5,0.5),             #Range of the speed
        ):
        self.weight = weight
        self.lr = lr
        self.max_iter = maxiter
        self.pop_size = popsize
        self.dim = dim
        self.poprange = poprange
        self.speedrange = speedrange

        self.gbest_history = np.empty(shape=[0, 1])
        self.gbest_pos_history = np.empty(shape=[0, self.dim])

    def init_particles(self):
        self.pop = np.empty(shape=[0, self.dim])
        self.speed = np.empty(shape=[0, self.dim])
        self.fitness = np.empty(shape=[0, 1])
        for i in range(self.pop_size):
            self.pop=np.append(self.pop,[np.random.uniform(self.poprange[0],self.poprange[1],self.dim)],axis=0)
            self.speed=np.append(self.speed,[np.random.uniform(self.speedrange[0],self.speedrange[1],self.dim)],axis=0)      
            self.fitness=np.append(self.fitness,[self.fitness_func(self.pop[i])])
    
    def init_gbest_pbest(self):
        self.gbest = self.pop[np.argmax(self.fitness)].copy()
        self.gbest_fitness = np.max(self.fitness).copy()
        self.pbest = self.pop.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_history = np.append(self.gbest_history,self.gbest_fitness)
        self.gbest_pos_history = np.append(self.gbest_pos_history,self.gbest)

    def fitness_func(self,x):
        if (x[0]==0)&(x[1]==0):
            y = np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
        else:
            y = np.sin(np.sqrt(x[0]**2+x[1]**2))/np.sqrt(x[0]**2+x[1]**2)+np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
        return y

    def update_gbest(self):
        t=1
        v = np.empty(shape=[self.pop_size, self.dim])
        #Update velocity
        for j in range(self.pop_size):
            v[j] += self.lr[0]*np.random.rand()*(self.pbest[j]-self.pop[j])+self.lr[1]*np.random.rand()*(self.gbest-self.pop[j])
        v[v<self.speedrange[0]] = self.speedrange[0]
        v[v>self.speedrange[1]] = self.speedrange[1]

        #Update position of particles
        for j in range(self.pop_size):
            self.pop[j] = t*(v[j])+(1-t)*self.pop[j]
        self.pop[self.pop<self.poprange[0]] = self.poprange[0]
        self.pop[self.pop>self.poprange[1]] = self.poprange[1]

        #Update fitness of particles
        for j in range(self.pop_size):
            self.fitness[j] = self.fitness_func(self.pop[j])
            if self.fitness[j]>self.pbest_fitness[j]:
                self.pbest[j] = self.pop[j].copy()
                self.pbest_fitness[j] = self.fitness[j].copy()
        
        #Update global best
        if self.pbest_fitness.max()>self.gbest_fitness:
            self.gbest = self.pop[np.argmax(self.pbest_fitness)].copy()
            self.gbest_fitness = self.pbest_fitness.max().copy()
            self.gbest_history = np.append(self.gbest_history,self.gbest_fitness)
            self.gbest_pos_history = np.append(self.gbest_pos_history,self.gbest,axis=0)

    def plot_function(self):
        fig = plt.figure() 
        ax = plt.axes(projection='3d')

        xx = np.arange(-5,5,0.1)
        yy = np.arange(-5,5,0.1)
        X, Y = np.meshgrid(xx, yy)
        Z = np.empty(shape=[len(xx),len(yy)])
        for i in range(len(xx)):
            for j in range(len(yy)):
                x = np.array([xx[i],yy[j]])
                Z[i,j] = self.fitness_func(x)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        plt.show()

    def run(self):
        self.init_particles()
        self.init_gbest_pbest()
        for i in range(self.max_iter):
            self.update_gbest()
        return self.gbest_history, self.gbest_pos_history

if __name__ == "__main__":
    pso = PSO()
    gbest_history, gbest_pos_history = pso.run()
    # print(gbest_history)
    # print(gbest_pos_history)
    pso.plot_function()
    plt.figure()
    plt.plot(gbest_history)
    plt.show()



