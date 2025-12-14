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
#' @param prior desc
#' @param prior_intercept desc
#' @param prior_phi desc
#' @param num_burnin_steps number of burn-in steps (currently these samples are all discarded)
#' @param num_results number of samples from each chain for each parameter after burn-in
#' @param n_chains number of MCMC chains (run in parallel if possible)
#' @param inits starting guess in simple MLE optimizer, which is used to generate starting position for the chains. Default is one value, and this same value is used as staring guess for all parameters. If a vector is passed which is of the same length of the number of parameters, then this is used as starting guess in the optimizer.
#' @param custompriors will be used to provide custom prior via a string - not implemented yet
#' @returns a list of lists. First list is a list of matrices of MCMC output, one member of the list for each estimated parameter, and each column in the matrix is the output of one chain. The number of columns in the matrix is the number of chains used. This is currently hardcoded. See function source. The second list is the python script sent to tensorflow, use cat() to view or write to file.
#' @examples
#' \dontrun{
#' library(rstanarm)
#' data(roaches)
#' roaches$roach1<-roaches$roach1/100;# manual
#' results_list<-glm_negbin(y ~ roach1 + treatment + senior + offset(log(exposure2)),
#' data = roaches, num_results=10000,num_burnin_steps=5000,n_chains=3)
#' samples<-results_list$samples
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
#' par(mfrow=c(1,2))
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
#'  phi_m<-samples[[3]] # the third parameter in the model, a matrix
#'  plot(density(c(phi_m)),col="skyblue",lwd=2, main="rstanarm (orange) v TF (blue)",
#'  xlab="treatment effect") # all chains combined
#'  lines(density(res_m[,"treatment"]),col="orange",lwd=2)
#'
#'
#'
#' }

#' @export
glm_negbin<-function(formula=NULL,data=NULL,prior=list(loc=0.,scale=2.5),
                     prior_intercept=list(loc=0.,scale=5.),
                     prior_phi=list(rate=1.0),
                     num_burnin_steps=5000,
                     num_results=20000,
                     n_chains=3,
                     inits=0.1,
                      custompriors=NULL) {

  # Extract the variables needed, including offset function, from the data
  # and make it available to python. Order of cols is important
  mf <- model.frame(formula=formula, data = data) # pass including offset
  # e.g. y~a+b+offset(log(z)) and this gives df with all relevant variables
  py$data<-r_to_py(mf) # this is needed as explicitly passes argument into py

  #mypriorstring<-buildNBpriorstr(formula);
  mystrs<-buildNBstr(mf)
  str1<-mystrs$str1
  str2<-mystrs$str2
  str3<-mystrs$str3
  str4<-mystrs$str4
  str5<-mystrs$str5
  str6<-mystrs$str6

  mypriorsstring<-buildNBpriorstr(mf,prior,prior_intercept,prior_phi)

  myinitsstring<-buildNBinits(mf,inits)

  # first part of py script - sets libraries and organizes the data passed
  stringpart1<-r"(

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tfb = tfp.bijectors
import numpy as np
import pandas as pd
import time

# data below is passed via R reticulate e.g., py$data<-r_to_py(model.frame)
# to run this as standalone script needs data read-in, e.g.
# data=pd.read_csv('data.csv')

rows, columns = data.shape

y_data=tf.convert_to_tensor(data.iloc[:,0], dtype = tf.float32)

X=tf.convert_to_tensor(data.iloc[:,1:],dtype=tf.float32)
X=tf.concat([tf.ones([rows,1],dtype=tf.float32), X], axis=1)

beta_expos=tf.convert_to_tensor(1.0,dtype=tf.float32) # dummy

)"

  stringpart2<-glue('
def make_observed_dist({str1}):
    """Function to create the observed Normal distribution."""
    B=tf.stack([{str2}])
    #print(B._shape_as_list())
    logmu=tf.linalg.matvec(X,B)

    phi_expanded = tf.expand_dims(phi, -1)  # (3, 1)
    mu = tf.math.exp(logmu)
    prob = phi_expanded/(phi_expanded+mu)

    # Create distribution for observations
    return(tfd.Independent(
        tfd.NegativeBinomial(total_count = phi_expanded, probs = 1-prob),
        reinterpreted_batch_ndims = 1
    ))

',.trim=FALSE)

  stringpart3<-r"(
model = tfd.JointDistributionSequentialAutoBatched([
)"

  ## mypriorsstring in HERE
  ## concat strings so far
  cumstring<-paste(stringpart1,stringpart2,stringpart3,mypriorsstring,sep="")

  stringpart4<-r"(
  make_observed_dist
])

tf.random.set_seed(99999)

)"

  stringpart5<-glue('
def log_prob_fn({str3}):
  """Unnormalized target density as a function of states."""
  return model.log_prob((
    {str3}, y_data))

  ',.trim=FALSE)

  stringpart6<-glue('
def neg_log_prob_fn(pars):
{str4}
   """Unnormalized target density as a function of states."""
   return -model.log_prob((
      {str3}, y_data))

',.trim=FALSE)

  stringpart7<-glue('

#### get starting values by find MLE
if(True):
    start = tf.constant([{myinitsstring}],dtype = tf.float32)  # Starting point for the search.
    optim_results = tfp.optimizer.nelder_mead_minimize(neg_log_prob_fn,
                 initial_vertex=start, func_tolerance=1e-04,max_iterations=1000)

    #print(optim_results.initial_objective_values)
    #print(optim_results.objective_value)
    #print(optim_results.position)

# bijector to map contrained parameters to real
unconstraining_bijectors = [
{str5}

]

num_results={num_results}
num_burnin_steps={num_burnin_steps}

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

n_chains={n_chains}
current_state = [
{str6}
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
samples = do_sampling()
t1 = time.time()
',.trim=FALSE)

stringpart8<-r"(

print("InferencE ran in {:.2f}s.".format(t1-t0))

samples = list(map(lambda x: tf.squeeze(x).numpy(), samples))
)"

bigstring<-paste(cumstring,stringpart4,stringpart5,stringpart6,stringpart7,stringpart8,sep="")
#return(bigstring)
py_run_string(bigstring)

return(list(samples=py_to_r(py$samples),py_code=bigstring))
}
