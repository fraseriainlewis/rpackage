# rpackage

library(rstanarm)
data(roaches)
roaches$roach1<-roaches$roach1/100;# manual
glm_negbin(thedata=roaches)
