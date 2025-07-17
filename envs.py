import numpy as np
from ruamel.yaml import YAML
from scipy.stats import skewnorm

class LinearEnv:

    def __init__(self, env_cfg) -> None:

        self.cfg = env_cfg
        self.A = np.array(env_cfg['A'])
        self.B = np.array(env_cfg['B'])
        self.C = np.array(env_cfg['C'])
        self.Q = np.diag(env_cfg['Q'])
        self.R = np.diag(env_cfg['R'])
        self.u_max = env_cfg['u_max']
        self.u_min = env_cfg['u_min']
        self.x_max = env_cfg['x_max']
        self.x_min = env_cfg['x_min']
        self.noise_cfg = env_cfg['noise']
        self.constraint_cfg = env_cfg['constraints']
        self.dt = env_cfg['dt']

        self.sample_count = 0

        pass

    def sample_noise(self, noise_cfg, shape):

        # apply time variant noise
        param = noise_cfg['param']
        if self.cfg['noise']['time_variant']['activate']:
            interval = self.cfg['noise']['time_variant']['interval']
            scale_factor = self.cfg['noise']['time_variant']['scale_factor']
            scale = np.ones(shape)
            scale = scale * (1 + scale_factor) * 0.3

            if self.sample_count % (2*interval) >= interval:
                scale = np.ones(shape)
                scale[-1] = scale_factor
            if scale_factor > 1:
                raise ValueError("scale factor should be less than 1")
        self.sample_count += 1

        

        # sample disturbance noise
        if noise_cfg['type'] == 'gaussian':
            n = np.random.normal(0, param, shape)
        elif noise_cfg['type'] == 'uniform':
            n = np.random.uniform(-param, param, shape)
        elif noise_cfg['type'] == 'triangular':
            n = np.random.triangular(-param, noise_cfg['skew'], param, shape)
            mean = (noise_cfg['skew']) / 3
            n -=  mean
        elif noise_cfg['type'] == 'bounded_laplace':
            # make it a bounded noise
            n = np.random.laplace(scale=param, size=shape)
            while np.linalg.norm(n)>noise_cfg['max_bound']:
                n = np.random.laplace(scale=param, size=shape)
        elif noise_cfg['type'] == 'gumbel':
            # make it a bounded noise
            n = np.random.gumbel(scale=param, size=shape)
            while np.linalg.norm(n)>noise_cfg['max_bound']:
                n = np.random.gumbel(scale=param, size=shape)

        elif noise_cfg['type'] == 'rayleigh':
            # make it a bounded noise
            n = np.random.rayleigh(scale=param, size=shape)
            while np.linalg.norm(n)>noise_cfg['max_bound']:
                n = np.random.rayleigh(scale=param, size=shape)

        elif noise_cfg['type'] == 'standard_cauchy':
            # make it a bounded noise
            n = np.random.standard_cauchy(size=shape)
            while np.linalg.norm(n)>noise_cfg['max_bound']:
                n = np.random.standard_cauchy(size=shape)
            n *= param

        elif noise_cfg['type'] == 'standard_t':
            # make it a bounded noise
            n = np.random.standard_t(df=5, size=shape)
            while np.linalg.norm(n)>noise_cfg['max_bound']:
                n = np.random.standard_t(df=5, size=shape)
            n *= param
        elif noise_cfg['type'] == 'star':
            if isinstance(shape, int) or len(shape) == 1:
                sample = np.random.laplace(scale=param, size=shape)
                while (np.abs(sample) > noise_cfg['star_bound']).all():
                    sample = np.random.laplace(scale=param, size=shape)
                n = sample
            else:
                samples = []
                for i in range(shape[0]):
                    sample = np.random.laplace(scale=param, size=shape[1:])
                    while (np.abs(sample) > noise_cfg['star_bound']).all():
                        sample = np.random.laplace(scale=param, size=shape[1:])
                    samples.append(sample)
                n = np.array(samples).reshape(shape)
        elif noise_cfg['type'] == 'skew_normal':
            n = skewnorm.rvs(noise_cfg['skew'], scale=param, size=shape)
            delta = noise_cfg['skew'] / np.sqrt(1 + noise_cfg['skew']**2)
            mean = param * delta * np.sqrt(2 / np.pi)
            n -= mean

        if self.cfg['noise']['time_variant']['activate']:
            n = np.multiply(n, scale)

        return n


    def reset(self, mean_x0):
            
        if mean_x0.shape[0] != self.A.shape[1]:
            raise ValueError("mean_x0 and A have incompatible shapes")

        x0 = mean_x0

        # sample init state noise
        init_noise = self.sample_noise(self.noise_cfg['init_state'], self.A.shape[0])

        x0 = x0 + init_noise

        # sample init obs noise
        init_obs_noise = self.sample_noise(self.noise_cfg['measurement'], self.C.shape[0])

        y0 = self.C @ x0 + init_obs_noise

        return x0, y0


    def step(self, x, u):

        if x.shape[0] != self.A.shape[1]:
            raise ValueError("x and A have incompatible shapes")
        
        if u.shape[0] != self.B.shape[1]:
            raise ValueError("u and B have incompatible shapes")
        
        x = self.A @ x + self.B @ u
        y = self.C @ x
        
        # sample disturbance noise
        w = self.sample_noise(self.noise_cfg['disturbance'], x.shape)

        # sample measurement noise
        e = self.sample_noise(self.noise_cfg['measurement'], y.shape)

        if 'state_variant' in self.noise_cfg:
            if x.shape[-1]==2 and x[1] > self.noise_cfg['state_variant']['y_threshold']:
                w = np.multiply(w, self.noise_cfg['state_variant']['above_scale'])
                e = np.multiply(e, self.noise_cfg['state_variant']['above_scale'])

        # apply noises
        x += w
        y += e
        
        return x, y
    
    def check_constraint(self, x):
        '''
        check if the state satisfies the constraint
        '''
        if self.constraint_cfg['circle'] == True:
            for i in range(len(self.constraint_cfg['mean'])):
                center = np.array(self.constraint_cfg['mean'][i])
                value = (
                    1.0 
                    - (x - center).reshape((1, -1))
                    @ np.linalg.inv(self.constraint_cfg['cov'][i])
                    @ (x - center).reshape((-1, 1))
                )
                if value > 0:
                    return False
        if self.constraint_cfg['polygon'] == True:
            for i in range(len(self.constraint_cfg['a'])):
                a = np.array(self.constraint_cfg['a'][i])
                b = np.array(self.constraint_cfg['b'][i])
                value = (np.dot(a, x) - b > 0).any()
                if value:
                    return False
        if self.constraint_cfg['exponential'] == True:
            shift = self.constraint_cfg['shift']
            radius = self.constraint_cfg['radius']
            value_1 = (x[1] - radius) - np.exp(-(x[0] - shift)) > 0
            # value_2 = (-x[1] - radius) - np.exp(-(x[0] - shift)) > 0
            if value_1:
                return False
        
        return True
    
