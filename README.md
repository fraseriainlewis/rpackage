# rpackage

```r
library(rtfp)
library(rstanarm)
data(roaches)
roaches$roach1<-roaches$roach1/100;# manual
samples<-glm_negbin(thedata=roaches) # this is from rtfp
```
