

#' Extracts specified variables from a dataset based on a model formula.
#' @param mf mode.frame
#' @return A list of strings which be glued into python string
buildNBstr<-function(mf){
  #return strings needed for negbin tfp model based on formula
  varnames<-names(mf)[-1] #drop response
  noms<-varnames[-grep("offset",varnames)] #drop offset
  Xs<-paste("beta_",noms,sep="") # e.g. alpha, beta_roach,beta_treatment,beta_senior,phi

  #print(Xs)
  str1<-c("phi",Xs[length(Xs):1],"alpha")
  str2<-c("alpha",Xs,"beta_expos")
  str3<-str1[length(str1):1]
  # phi, beta_senior,beta_treatment, beta_roach,alpha
  # alpha,beta_roach,beta_treatment,beta_senior,beta_expos
  # alpha, beta_roach,beta_treatment,beta_senior,phi

  str4<-paste(paste(paste(paste("   ",str3,sep=""),"=pars[[",sep=""),0:(length(str3)-1),sep=""),"]]",sep="")
  #alpha=pars[[0]]
  #beta_roach1=pars[[1]]
  #beta_treatment=pars[[2]]
  #beta_senior=pars[[3]]
  #phi=pars[[4]]

  str5<-paste("   ",c(rep("tfb.Identity(),",length(str3)-1),"tfb.Exp()"),collapse="\n")
  #tfb.Identity(),
  #tfb.Identity(),
  #tfb.Identity(),
  #tfb.Identity(),
  #tfb.Exp()

  return(list(str1=paste(str1,collapse=","),
              str2=paste(str2,collapse=","),
              str3=paste(str3,collapse=","),
              str4=paste(str4,collapse="\n"),
              str5=str5))


}


#' Extracts specified variables from a dataset based on a model formula.
#' @param mf mode.frame
#' @return A list of strings which be glued into python string
buildNBpriorstr<-function(mf){
  #return strings needed for negbin tfp model based on formula
  varnames<-names(mf)[-1] #drop response
  noms<-varnames[-grep("offset",varnames)] #drop offset
  Xs<-paste("beta_",noms,sep="") # e.g. alpha, beta_roach,beta_treatment,beta_senior,phi

  str1<-c("phi",Xs[length(Xs):1],"alpha")
  str<-str1[length(str1):1]
  # alpha, beta_roach,beta_treatment,beta_senior,phi

  #
  #tfd.Normal(loc=0., scale=5., name="alpha"),
  #tfd.Normal(loc=0., scale=2.5, name="beta_roach1"),
  #tfd.Normal(loc=0., scale=2.5, name="beta_treatment"),
  #tfd.Normal(loc=0., scale=2.5, name="beta_senior"),
  #tfd.Exponential(rate=1., name="phi"))"

}


