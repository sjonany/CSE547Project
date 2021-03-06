#setwd("C:/Users/Jerry/CSE547FinalProj/CSE547Project/src/util/MCMC")

######################################
# HYPERPARAMETER INITIALIZATION
######################################
T <- 100 # number of topics
#T <- 10
gamma <- 0.01 # used to smooth C^{WT}
alpha <- 50 / T # used to smooth C^{DT}

#burnin <- 750 # number of steps to advance the Markov chain before start taking samples
burnin <- 0
#numIter <- 1000 # total number of iterations (including burn-in iterations) 
numIter <- 20
lag <- 1 # will take sample every lag number of iterations 
numSampsPerLag = 1 # number of consecutive iterations to sample from 
totNumSamps = 0 # will be incremented as new samples come

######################################
# DATA INGESTION
######################################
#N = number of distinct nouns in corpus
#V = number of distinct verbs in corpus

verbIdx.dat <- read.table("verbIdx.txt")
# verbIdx.txt lists all the unique verbs
verbIdx <- as.vector(verbIdx.dat) # a one by V vector

V <- dim(verbIdx)[1]

nounIdx.dat <- read.table("nounIdx.txt")
# nounIdx.txt lists all the unique nouns
nounIdx <- as.vector(nounIdx.dat) # a one by N vector

N <- dim(nounIdx)[1]

#vnIdx.dat <- read.csv(header = TRUE, "vnIdx.txt")
# vnIdx lists all the (vIdx,nIdx) instances (with duplicates), sorted first by vIdx then by nIdx
#vnIdx <- as.matrix(vnIdx.dat) # a one by |corpus| vector

#C <- dim(vnIdx)[1]
C <- 6594702 # total number of tuples in vnIdx
# NOTE: vnIdx test file must have an empty line at the end
print(sprintf("start training with %d tuples in vnIdx...", C))
######################################
# MARKOV CHAIN INITIALIZATION
######################################
#For each (v,n) pair with index i, z[i] will store an int from 1 to T, telling us the current topic assignment of this pair
z <- rep(0, C)

# cumBeta and cumTheta will store the accumulated distro vectors, and be used to average
# (in the sense of Monte Carlo) once the Markov Chain stabilizes
cumBeta <- matrix(0, T, N)
cumTheta <- matrix(0, V, T)

#C^{WT}_{n,t} tells us the number of times a noun n has been assigned to topic t
CWT <- matrix(0, N, T)

#C^{VT}_{v,t} tells us the number of times a topic t has been assigned to the nouns that ever appear in a (v,n) pair
CVT <- matrix(0, V, T)

# tells us the number of nouns that have been assigned to each topic (i.e. the column sum of CWT)
numNounsInTopic <- rep(0, T)

# tells us the number of nouns associated with each verb (i.e. the row sum of CVT)
numNounsWithVerb <- rep(0, V)

# file connection for the vnIdx file
vnIdxCon <- file("vnIdxSmall.txt", "r")

for (i in 1:C) {
  
  
  t <- sample(T, 1) # randomly choose t = 1 to T
  z[i] <- t
  
  if(i %% 10000 == 0) print(sprintf("assigning tuple %d to topic %d", i, t))
  #curV <- vnIdx[i,1]
  #curN <- vnIdx[i,2]
  
  nextLine <- readLines(vnIdxCon, ok=FALSE, 1)
  vnPair = as.numeric(unlist(strsplit(nextLine, ",")))
  curV <- vnPair[1]
  curN <- vnPair[2]
  
  #print(sprintf("curV = %d, curN=%d", curV, curN))
  CWT[curN,t] = CWT[curN,t] + 1
  CVT[curV,t] = CVT[curV,t] + 1
  
  numNounsInTopic[t] = numNounsInTopic[t] + 1
  numNounsWithVerb[curV] = numNounsWithVerb[curV] + 1
}
close(vnIdxCon)
#print(CWT)
#print(CVT)

######################################
# COLLAPSED GIBBS SAMPLING
######################################
for (iter in 1:numIter) {
  print(sprintf("At iteration %d", iter))
  
  # file connection for the vnIdx file
  vnIdxCon <- file("vnIdxSmall.txt", "r")
  
  for (i in 1:C) {

    tau <- z[i] #  the current topic assignment for this pair
    #curV <- vnIdx[i,1]
    #curN <- vnIdx[i,2]
    nextLine <- readLines(vnIdxCon, ok=FALSE, 1)
    vnPair = as.numeric(unlist(strsplit(nextLine, ",")))
    curV <- vnPair[1]
    curN <- vnPair[2]
    
    #print(sprintf("current topic id for (%d,%d) is %d", curV, curN, tau))
    #print(CVT)
    #print(CWT)
    
    if (CWT[curN,tau] == 0) print(sprintf("warning: subtracting zero entry CWT[%d,%d]", curN, tau))
    CWT[curN,tau] = CWT[curN,tau] - 1
  
    if (CVT[curV,tau] == 0) print(sprintf("warning: subtracting zero entry CVT[%d,%d]", curV, tau))
    CVT[curV,tau] = CVT[curV,tau] - 1
    
    numNounsInTopic[tau] = numNounsInTopic[tau] - 1
    numNounsWithVerb[curV] = numNounsWithVerb[curV] - 1
    
    resampleDistro <- rep(0, T)
    
    for (t in 1:T) {
      pnt <- (CWT[curN,t] + gamma) / (numNounsInTopic[t] + N * gamma)
      ptv <- (CVT[curV,t] + alpha) / (numNounsWithVerb[curV] + T * alpha)
      resampleDistro[t] <- pnt * ptv
    }

    #print(resampleDistro)
    
    tau <- sample(T, 1, prob = resampleDistro)

    #print(sprintf("resampling topic id for (%d,%d) to %d", curV, curN, tau))
    CWT[curN,tau] = CWT[curN,tau] + 1
    CVT[curV,tau] = CVT[curV,tau] + 1
    
    numNounsInTopic[tau] = numNounsInTopic[tau] + 1
    numNounsWithVerb[curV] = numNounsWithVerb[curV] + 1
    
    z[i] <- tau
    #print(CVT)
    #print(CWT)
  }
  
  if (iter > burnin && (iter %% lag) < numSampsPerLag) {
    # only sample when iter = k*lag, k*lag+1, ..., k * lag + (numSampsPerLag - 1)
    for (t in 1:T) {
      for (n in 1:N) {
        cumBeta[t,n] = cumBeta[t,n] + (CWT[n,t] + gamma) / (numNounsInTopic[t] + N * gamma)
      }
    }
    
    for (v in 1:V) {
      for (t in 1:T) {
        cumTheta[v,t] = cumTheta[v,t] + (CVT[v,t] + alpha) / (numNounsWithVerb[v] + T * alpha)
      }
    }
    
    totNumSamps = totNumSamps + 1
  }  
  close(vnIdxCon)
}

###############################################
# COMPUTE THE VERB_TOPIC AND TOPIC_NOUN DISTROS
###############################################
# beta_j is a distribution over the N nouns, theta_r is a distribution over the T topics
beta <- cumBeta / totNumSamps

theta <- cumTheta / totNumSamps

#write(t(beta), "beta.txt", ncolumns=N, sep = "\t")
#write(t(theta), "theta.txt", ncolumns=T, sep = "\t")

###############################################
# TESTS
###############################################

#############################################
# Gives the probability p(NOT D | v,n)
# PARAMS:
#   vIdx: the numerical index of query verb v
#   nIdx: the numerical index of query noun n
#   pNotD: p(not D) (computed from the training set)
#   pnd: p(n | D) (computed from the training set)
#############################################
givePlausibility <- function(vIdx, nIdxm, pNotD, pnd) {
  plda_nv <- 0.0
  for (t in 1:T) {
    plda_nv <- plda_nv + beta[t,n] * theta[v,t]
  }
  
  ans = (pNotD * plda_nv) / ((1-pNotD) * pnd + pNotD * plda_nv)
  return (ans)
}