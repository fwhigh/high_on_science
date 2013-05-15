---
published: true
layout: post
title: Solve a system of linear equations in R
date: "2013-05-15T11:13:29.000Z"
categories: R

---

Allegedly, the following puzzle is easily solved by young children in a matter of minutes, meanwhile reducing the average highly educated adult to shame and tears.

{% highlight R %}
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
{% endhighlight %}
