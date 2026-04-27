import numpy as np
from CortexReconstructionMnist2 import NeuralEncoder




class NeuralDecoder(NeuralEncoder):
    def __init__(self):
        self.particles=None
        self.predx=None
        self.predy=None
    def fit(self,ND):
        self.__dict__.update(ND.__dict__)
    def _init_particles(self,n_particles=300):
        self.n_particles=n_particles
        self.particles=(1/n_particles)*np.ones(size=(n_particles,))
        self.predx=np.zeros(size=(n_particles,))
        self.predx=np.zeros(size=(n_particles,))

    def _propagate(self,curr_):
        for idx in range(self.n_particles):
            self.predx[idx]=self.ganglion_x[curr_]+np.random.normal(0,np.sqrt(self.D * self.dt))
            self.predy[idx]=self.ganglion_y[curr_]+np.random.normal(0,np.sqrt(self.D * self.dt))
    def _reweight(self):
        pass