import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tfb = tfp.bijectors
import numpy as np
import pandas as pd
import time

data=pd.read_csv('roaches.csv')
data.roach1=data.roach1/100;# manual fix special case - should be done in passed data

rows, columns = data.shape

y_data=tf.convert_to_tensor(data.iloc[:,0], dtype = tf.float32)

#roach_data=tf.convert_to_tensor(data.iloc[:,1], dtype = tf.float32)
#trt_data=tf.convert_to_tensor(data.iloc[:,2], dtype = tf.float32)
#snr_data=tf.convert_to_tensor(data.iloc[:,3], dtype = tf.float32)

X=tf.convert_to_tensor(data.iloc[:,1:],dtype=tf.float32)
X=tf.concat([tf.ones([rows,1],dtype=tf.float32), X], axis=1)

#exposure_data=tf.math.log(tf.expand_dims(tf.convert_to_tensor(data.iloc[:,4],dtype=tf.float32),axis=-1))

#X=tf.concat([X,exposure_data], axis=1)

beta_expos=tf.convert_to_tensor(1.0,dtype=tf.float32) # dummy

def make_observed_dist(phi, beta_senior,beta_treatment, beta_roach,alpha):
    """Function to create the observed Normal distribution."""
    
    B=tf.stack([alpha,beta_roach,beta_treatment,beta_senior,beta_expos])
    #print(B._shape_as_list())
    logmu=tf.linalg.matvec(X,B)

    phi_expanded = tf.expand_dims(phi, -1)  # (3, 1)
    mu = tf.math.exp(logmu)
    #r = tf$expand_dims(1.0, -1L)/ phi_expanded;  # total_count = 5
    #probs <- r / (r + mu)

    prob = phi_expanded/(phi_expanded+mu)
    #prob<-(phi_expanded)/(mu+phi_expanded)
    # Create distribution for observations
    return(tfd.Independent(
        #tfd_normal(loc = mu, scale = sigma_expanded),
        #mu = exp(mu)
        #phi <- 0.2  # scale/overdispersion

        #r = tf$expand_dims(1.0, -1L)/ sigma_expanded;  # total_count = 5
        #probs <- r / (r + mu)
        tfd.NegativeBinomial(total_count = phi_expanded, probs = 1-prob),
        reinterpreted_batch_ndims = 1
    ))

## -------- this needs contructed via cat
# Define the joint distribution without matrix mult
model = tfd.JointDistributionSequentialAutoBatched([
  tfd.Normal(loc=0., scale=5., name="alpha"),  # # Intercept (alpha)
  tfd.Normal(loc=0., scale=2.5, name="beta_roach"),  # # Slope (beta_roach1)
  tfd.Normal(loc=0., scale=2.5, name="beta_treatment"),  # # Intercept (alpha)
  tfd.Normal(loc=0., scale=2.5, name="beta_senior"),  # # Slope (beta_roach1)
  tfd.Exponential(rate=1., name="phi"),
  make_observed_dist
])

tf.random.set_seed(99999)

def log_prob_fn(alpha, beta_roach,beta_treatment,beta_senior,phi):
  """Unnormalized target density as a function of states."""
  return model.log_prob((
      alpha, beta_roach,beta_treatment,beta_senior,phi, y_data))

def neg_log_prob_fn(pars):
    alpha=pars[[0]]
    beta_roach=pars[[1]]
    beta_treatment=pars[[2]]
    beta_senior=pars[[3]]
    phi=pars[[4]]
    """Unnormalized target density as a function of states."""
    return -model.log_prob((
      alpha, beta_roach,beta_treatment,beta_senior,phi, y_data))


#print(log_prob_fn(0.1,0.2,0.3,0.5,0.1))
start = tf.constant([0.1,0.2,0.3,0.5,0.1],dtype = tf.float32)
#print(neg_log_prob_fn(start))

#### get starting values by find MLE
if(True):
    start = tf.constant([0.1,0.2,0.3,0.5,0.1],dtype = tf.float32)  # Starting point for the search.
    optim_results = tfp.optimizer.nelder_mead_minimize(neg_log_prob_fn,
                 initial_vertex=start, func_tolerance=1e-04,max_iterations=1000)

    #print(optim_results.initial_objective_values)
    #print(optim_results.objective_value)
    #print(optim_results.position)

# bijector to map contrained parameters to real
unconstraining_bijectors = [
    tfb.Identity(),
    tfb.Identity(),
    tfb.Identity(),
    tfb.Identity(),
    tfb.Exp()
]

num_results=25000
num_burnin_steps=5000

sampler = tfp.mcmc.TransformedTransitionKernel(
    tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_fn,
        step_size=tf.cast(0.5, tf.float32)), #tf.cast(0.1, tf.float32)),
    bijector=unconstraining_bijectors
    )

adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=sampler,
    num_adaptation_steps=int(0.8 * num_burnin_steps),
    target_accept_prob=tf.cast(0.75, tf.float32))

istate = optim_results.position

n_chains=3
current_state = [tf.expand_dims(tf.repeat(istate[0],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[1],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[2],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[3],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[4],repeats=n_chains,axis=-1),axis=-1)
                 ]

# Speed up sampling by tracing with `tf.function`.
@tf.function(autograph=False, jit_compile=True,reduce_retracing=True)
def do_sampling():
  return tfp.mcmc.sample_chain(
      kernel=adaptive_sampler,
      current_state=current_state,
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      trace_fn=None)#lambda current_state, kernel_results: kernel_results)


t0 = time.time()
#samples, kernel_results = do_sampling()
samples = do_sampling()
t1 = time.time()
print("Inference ran in {:.2f}s.".format(t1-t0))

samples = list(map(lambda x: tf.squeeze(x).numpy(), samples))

print(samples)
#print(kernel_results.shape)




