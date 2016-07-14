library(boot)

n.boot <- 1000
n.points <- 1000
rnorm.mean <- 0
rnorm.sd <- 1
query.val <- rnorm.mean - 3*rnorm.sd

rank.fun <- function(data, indices, query) {
  d <- data[indices,]
  over.under <- query > d$value
  val <- sum(over.under)
  return(val)
}

data <- data.frame(label=1:n.points,value=rnorm(n.points, mean=rnorm.mean, sd=rnorm.sd))

ptm <- proc.time()
results <- boot(data=data, statistic=rank.fun, R=n.boot, query=query.val)
print(proc.time() - ptm)

plot(results, index=1) # intercept 

print(results)
print(boot.ci(results, type="bca", index=1))
