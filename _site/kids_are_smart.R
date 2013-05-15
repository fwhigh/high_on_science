# http://highonscience.com/2013/02/13/kids-are-smart/

### Library
library(limSolve)


### Functions

# Make the matrix A from the puzzle data table
training.to.matrix.A <- function(train)
{
    output<-matrix(nrow=nrow(train),ncol=10)
    for (i in 1:nrow(train))
    {
        ones      <- get.digit(train$Input[i],1e0)
        tens      <- get.digit(train$Input[i],1e1)
        hundreds  <- get.digit(train$Input[i],1e2)
        thousands <- get.digit(train$Input[i],1e3)

        row <- rep(0,10)

        row[ones+1]      <- row[ones+1]     +1
        row[tens+1]      <- row[tens+1]     +1
        row[hundreds+1]  <- row[hundreds+1] +1
        row[thousands+1] <- row[thousands+1]+1

        output[i,] <- row
    }
    return(output)
}

# Extract a digit from a number.
# Ones digit: set place to 1e0
# Tens digit: set place to 1e1
# Hundreds digit: set place to 1e2
# Etc.
get.digit <- function(number, place)
{
    digit <- floor((number %% (place*10))/place)
    return(digit)
}


### Main body

# Make a data frame out of the puzzle to serve as training set
training.input <- as.data.frame(c(8809,7111,2172,6666,1111,3213,7662,9313,0000,2222,3333,5555,8193,8096,7777,9999,7756,6855,9881,5531))
training.output <- as.data.frame(c(6,0,0,4,0,0,2,1,4,0,0,0,3,5,0,4,1,3,5,0))
training.data <- cbind(training.input,
                       training.output)
colnames(training.data) <- c("Input",
                             "Output")

# Make the matrix A and the column vector b
A <- training.to.matrix.A(training.data)
b <- matrix(training.data$Output)

# Solve A x = b ...
# ... using the package limSolve, dedicated to this purpose
fit <- Solve(A,b)
print(fit)

# ... using the engine behind lm
fit1<- lm.fit(A, b)
print(fit.alt$coefficients)

# Use lm directly
training.table <- as.data.frame(A)
training.table$Output <- b
fit2 <- lm(Output ~ 0 + ., data=training.table)
print(fit2$coefficients)
