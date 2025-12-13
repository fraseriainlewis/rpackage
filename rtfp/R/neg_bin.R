# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   https://r-pkgs.org
#
# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Cmd + Shift + B'
#   Check Package:             'Cmd + Shift + E'
#   Test Package:              'Cmd + Shift + T'

.onLoad <- function(libname, pkgname) {
  reticulate::py_require("tensorflow")
  reticulate::py_require("tf_keras")
  reticulate::py_require("tensorflow_probability")
}

#' Add together two numbers
#'
#' @returns The sum of `x` and `y`
#' @export
#' @examples
#'
#' test_call_to_tfd()
test_call_to_tfd <- function() {
  print("calling....\n")
  return(tfd_bernoulli(probs=0.5)%>%tfd_sample(5L))
}

#' Add together two numbers
#'
#' @returns The sum of `x` and `y`
#' @export
#' @examples
#'
#' test_call_to_tfd()
script_tfd <- function() {
  print("calling....\n")
  a<-10.0
  py$a<-a
  #loc_x<-x;
  #assign("loc_x", loc_x, envir = .GlobalEnv)
  #print(tfd_bernoulli(probs=0.5)%>%tfd_sample(5L))
  mystring<-"
import tensorflow as tf
import tensorflow_probability as tfd

print(a)
#fromr= r.a
#fromr2=r.roaches
#print(type(r.roaches))
#print(r.loc_x)
#print(fromr)
"
  py_run_string(mystring)
cat("got=",py$a,"\n")

# Clean up global environment
#rm(loc_x, envir = .GlobalEnv)

}



#' Fit Bayesian Negative Binomial Additive model using Tensorflow
#'
#' @description A short description here. See the example below, currently this function fits the same negative binomial regression model as given in the rstanarm example. The data set is passed and the rest of the function is currently hard coded to this example, using one of the tensorflow probability samplers. The function is written in python and called via reticulate which also brings the MCMC sample data into R. The code is simple starting point for expansion. The same function could be written in Rstudio tfprobability library, but needs considerable care on the broadcasting.
#' @param formula data.frame of the data to fit to the model - must be the dataset given in the example
#' @param data description
#' @param priors dddd
#' @returns a list of matrices of MCMC output, one member of the list for each estimated parameter, and each column in the matrix is the output of one chain. The number of columns in the matrix is the number of chains used. This is currently hardcoded. See function source.
#' @examples
#' \dontrun{
#' library(rstanarm)
#' data(roaches)
#' roaches$roach1<-roaches$roach1/100;# manual
#' samples<-glm_negbin(thedata=roaches)
#' ## Trace plots for the reciprocal dispersion parameter
#' phi_m<-samples[[5]] # the fifth parameter in the model, a matrix
#' par(mfrow=c(2,2))
#' plot(phi_m[,1],type="l",col="green",main="Trace plots")
#' lines(phi_m[,2],col="blue")
#' lines(phi_m[,3],col="skyblue")
#' plot(phi_m[,1],type="l",col="green",main="Trace plots")
#' plot(phi_m[,2],type="l",col="blue",main="Trace plots")
#' plot(phi_m[,3],type="l",col="skyblue",main="Trace plots")
#'
#' ## Density plots and compare with rstanarm
#' par(mfrow=c(1,1))
#' plot(density(c(phi_m)),col="skyblue",lwd=2, main="rstanarm (orange) v TF (blue)",
#' xlab="Reciprocal Dispersion") # all chains combined
#'
#' glm1 <- glm(y ~ roach1 + treatment + senior, offset = log(exposure2),
#' data = roaches, family = poisson)
#' stan_glm1 <- stan_glm(y ~ roach1 + treatment + senior, offset = log(exposure2),
#'                       data = roaches, family = neg_binomial_2,
#'                       prior = normal(0, 2.5),
#'                       prior_intercept = normal(0, 5),
#'                       seed = 12345,
#'                       warmup = 10000,      # Number of warmup iterations per chain
#'                       iter = 20000,        # Total iterations per chain (warmup + sampling)
#'                       thin = 1,
#'                       chains = 2)           # Thinning rate)
#'  res_m<-as.matrix(stan_glm1)
#'  lines(density(res_m[,"reciprocal_dispersion"]),col="orange",lwd=2)
#'
#'
#' }

#' @export
glm_negbin<-function(formula=NULL,data=NULL,
                      priors=NULL) {

  mypriorsstring<-r"(
  tfd.Normal(loc=0., scale=5., name="alpha"),
  tfd.Normal(loc=0., scale=2.5, name="beta_roach"),
  tfd.Normal(loc=0., scale=2.5, name="beta_treatment"),
  tfd.Normal(loc=0., scale=2.5, name="beta_senior"),
  tfd.Exponential(rate=1., name="phi"))"

  mf <- model.frame(formula=formula, data = data) # pass including offset
  # e.g. y~a+b+offset(log(z)) and this gives df with all relevant variables
  py$data<-r_to_py(mf) # this is needed as explicitly passes argument into py
  #print(py$data)
  #return(mf)

  stringpart1<-r"(

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tfb = tfp.bijectors
import numpy as np
import pandas as pd
import time

rows, columns = data.shape

y_data=tf.convert_to_tensor(data.iloc[:,0], dtype = tf.float32)
#roach_data=tf.convert_to_tensor(data.iloc[:,1], dtype = tf.float32)
#trt_data=tf.convert_to_tensor(data.iloc[:,2], dtype = tf.float32)
#snr_data=tf.convert_to_tensor(data.iloc[:,3], dtype = tf.float32)

X=tf.convert_to_tensor(data.iloc[:,1:],dtype=tf.float32)
#print(X)
#exit(1)
X=tf.concat([tf.ones([rows,1],dtype=tf.float32), X], axis=1)

#exposure_data=tf.math.log(tf.expand_dims(tf.convert_to_tensor(data.iloc[:,4],dtype=tf.float32),axis=-1))

#X=tf.concat([X,exposure_data], axis=1)

beta_expos=tf.convert_to_tensor(1.0,dtype=tf.float32) # dummy

)"

  stringpart2<-r"(
def make_observed_dist(phi, beta_senior,beta_treatment, beta_roach,alpha):
    """Function to create the observed Normal distribution."""
    #alpha_expanded = tf.expand_dims(alpha, -1)  # (3, 1)
    #beta_roach_expanded = tf.expand_dims(beta_roach, -1)    # (3, 1)
    #beta_treatment_expanded = tf.expand_dims(beta_treatment, -1)    # (3, 1)
    #beta_senior_expanded = tf.expand_dims(beta_senior, -1)    # (3, 1)
    #beta_expos_expanded = tf.expand_dims(1.0, -1)

    #logmu = alpha_expanded + beta_roach_expanded * roach_data + beta_treatment_expanded * trt_data + beta_senior_expanded * snr_data +tf.math.log(exposure_data)

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

)"

  stringpart3<-r"(
model = tfd.JointDistributionSequentialAutoBatched([
  )"

  #tfd.Normal(loc=0., scale=5., name="alpha"),  # # Intercept (alpha)
  #tfd.Normal(loc=0., scale=2.5, name="beta_roach"),  # # Slope (beta_roach1)
  #tfd.Normal(loc=0., scale=2.5, name="beta_treatment"),  # # Intercept (alpha)
  #tfd.Normal(loc=0., scale=2.5, name="beta_senior"),  # # Slope (beta_roach1)
  #tfd.Exponential(rate=1., name="phi"),

  ## mypriorsstring in HERE

  cumstring<-paste(stringpart1,stringpart2,stringpart3,mypriorsstring,",",sep="")

  stringpart4<-r"(
  make_observed_dist
])

tf.random.set_seed(99999)

)"

  varstring<-r"(alpha, beta_roach,beta_treatment,beta_senior,phi)"

  stringpart5<-glue('
def log_prob_fn({varstring}):
  """Unnormalized target density as a function of states."""
  return model.log_prob((
    {varstring}, y_data))

  ',.trim=FALSE)

  varstring2<-r"(
    alpha=pars[[0]]
    beta_roach=pars[[1]]
    beta_treatment=pars[[2]]
    beta_senior=pars[[3]]
    phi=pars[[4]])"

  stringpart6<-glue('
def neg_log_prob_fn(pars):{varstring2}
    """Unnormalized target density as a function of states."""
    return -model.log_prob((
      {varstring}, y_data))

',.trim=FALSE)

  stringpart7<-r"(
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

num_results=1000
num_burnin_steps=100

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

)"
bigstring<-paste(cumstring,stringpart4,stringpart5,stringpart6,stringpart7,sep="")
#return(bigstring)
py_run_string(bigstring)

return(py_to_r(py$samples))
}
