---
published: true
layout: post
comments: true
title: Solve a system of linear equations in R
date: "2013-05-15T11:13:29.000Z"
categories: R
---

Allegedly, the following puzzle is easily solved by young children in a matter of minutes, meanwhile reducing the average highly educated adult to shame and tears.

> 8809 = 6  
> 7111 = 0  
> 2172 = 0  
> 6666 = 4  
> 1111 = 0  
> 3213 = 0  
> 7662 = 2  
> 9313 = 1  
> 0000 = 4  
> 2222 = 0  
> 3333 = 0  
> 5555 = 0  
> 8193 = 3  
> 8096 = 5  
> 7777 = 0  
> 9999 = 4  
> 7756 = 1  
> 6855 = 3  
> 9881 = 5  
> 5531 = 0  
> 2581 = ?

I'm sure you're super smart and you solved it, but I don't care because your brilliance is not the point of this post.  This puzzle can be thought of as a system linear equation equations, and I'll use it to demonstrate solving such a system in R.

Casting the puzzle as a linear algebra problem
----------------------------------------------

The puzzle can be cast as a system of linear equations if you think of each digit on the left hand side as an unknown variable.

![Alt text]({{ site.baseurl }}/images/tex2img_1360772058.jpg "Optional title")

This effectively says, the puzzle involves counting something about the digits and adding them up to get the values on the right hand side.  

This is a matrix equation, **A** **x** = **b**, with _m_ = 20 and _n_ = 10.  _A<sub>ij</sub>_ is equal to the number of times the integer _x<sub>j</sub>_ appears on the left-hand side of a given line in the puzzle, which can be zero to four times given the four digit numbers.

Solving systems of linear equations is a well understood problem and is closely related to linear regression.  While R provides a base level routine called solve for this purpose, solve needs A to be a square, _n_ &times; _n_ matrix.  The puzzle clue represents a (presumably) overdetermined system of equations because _m_ is greater than _n_.  The library limSolve can handle this situation with its own Solve.

Solving it in R
---------------

Here's [the gist](https://gist.github.com/fwhigh/5602401 "Github gist for this post").  We first load the library and then define two subroutines that transform the numbers into useful data types.

{% highlight r %}
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

The first subroutine, `training.to.matrix.A`, converts a data table containing the puzzle clue, or training data, into the matrix **A**.  The second subroutine is needed by the first one—it extracts an individual digit from a number.

Now, to solve it.

{% highlight r %}
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
{% endhighlight %}


This gives the right answer, to machine precision.  

It turns out we could have used a function that works behind the scenes of the lm routine.

{% highlight r %}
# ... using the engine behind lm
fit1<- lm.fit(A, b)
print(fit1$coefficients)
{% endhighlight %}

Results are the same, but this function’s documentation comes with an explicit warning to use with caution.

What did we learn?
------------------

Mainly how to use another R package, `limSolve`, by casting a puzzle as an overdetermined system of linear equations.  It was easy to solve once we transformed the data into a useful format, and as is almost always the case, the transformations take up well over half the code while the solution itself takes up one line.

_[Credit: Cerebral Mastication](http://www.cerebralmastication.com/2012/03/solving-easy-problems-the-hard-way/)_
