---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,markdown//md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
---

# Introduction to Python

<a target="_blank" href="https://colab.research.google.com/github/intro-stat-learning/ISLP_labs/blob/v2.2/Ch02-statlearn-lab.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/intro-stat-learning/ISLP_labs/v2.2?labpath=Ch02-statlearn-lab.ipynb)






## Getting Started


To run the labs in this book, you will need two things:

* An installation of `Python3`, which is the specific version of `Python`  used in the labs. 
* Access to  `Jupyter`, a very popular `Python` interface that runs code through a file called a *notebook*. 


You can download and install  `Python3`   by following the instructions available at [anaconda.com](http://anaconda.com). 


 There are a number of ways to get access to `Jupyter`. Here are just a few:
 
 * Using Google's `Colaboratory` service: [colab.research.google.com/](https://colab.research.google.com/). 
 * Using `JupyterHub`, available at [jupyter.org/hub](https://jupyter.org/hub). 
 * Using your own `jupyter` installation. Installation instructions are available at [jupyter.org/install](https://jupyter.org/install). 
 
Please see the `Python` resources page on the book website [statlearning.com](https://www.statlearning.com) for up-to-date information about getting `Python` and `Jupyter` working on your computer. 

You will need to install the `ISLP` package, which provides access to the datasets and custom-built functions that we provide.
Inside a macOS or Linux terminal type `pip install ISLP`; this also installs most other packages needed in the labs. The `Python` resources page has a link to the `ISLP` documentation website.

To run this lab, download the file `Ch2-statlearn-lab.ipynb` from the `Python` resources page. 
Now run the following code at the command line: `jupyter lab Ch2-statlearn-lab.ipynb`.

If you're using Windows, you can use the `start menu` to access `anaconda`, and follow the links. For example, to install `ISLP` and run this lab, you can run the same code above in an `anaconda` shell.



## Basic Commands



In this lab, we will introduce some simple `Python` commands. 
 For more resources about `Python` in general, readers may want to consult the tutorial at [docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/). 


 



Like most programming languages, `Python` uses *functions*
to perform operations.   To run a
function called `fun`, we type
`fun(input1,input2)`, where the inputs (or *arguments*)
`input1` and `input2` tell
`Python` how to run the function.  A function can have any number of
inputs. For example, the
`print()`  function outputs a text representation of all of its arguments to the console.

```python
print("fit a model with", 11, "variables")

```

 The following command will provide information about the `print()` function.

```python
print?

```

Adding two integers in `Python` is pretty intuitive.

```python
3 + 5

```

In `Python`, textual data is handled using
*strings*. For instance, `"hello"` and
`'hello'`
are strings. 
We can concatenate them using the addition `+` symbol.

```python
"hello" + " " + "world"

```

 A string is actually a type of *sequence*: this is a generic term for an ordered list. 
 The three most important types of sequences are lists, tuples, and strings.  
We introduce lists now. 


The following command instructs `Python` to join together
the numbers 3, 4, and 5, and to save them as a
*list* named `x`. When we
type `x`, it gives us back the list.

```python
x = [3, 4, 5]
x

```

Note that we used the brackets
`[]` to construct this list. 

We will often want to add two sets of numbers together. It is reasonable to try the following code,
though it will not produce the desired results.

```python
y = [4, 9, 7]
x + y

```

The result may appear slightly counterintuitive: why did `Python` not add the entries of the lists
element-by-element? 
 In `Python`, lists hold *arbitrary* objects, and  are added using  *concatenation*. 
 In fact, concatenation is the behavior that we saw earlier when we entered `"hello" + " " + "world"`. 
 


This example reflects the fact that 
 `Python` is a general-purpose programming language. Much of `Python`'s  data-specific
functionality comes from other packages, notably `numpy`
and `pandas`. 
In the next section, we will introduce the  `numpy` package. 
See [docs.scipy.org/doc/numpy/user/quickstart.html](https://docs.scipy.org/doc/numpy/user/quickstart.html) for more information about `numpy`.



## Introduction to Numerical Python

As mentioned earlier, this book makes use of functionality   that is contained in the `numpy` 
 *library*, or *package*. A package is a collection of modules that are not necessarily included in 
 the base `Python` distribution. The name `numpy` is an abbreviation for *numerical Python*. 


  To access `numpy`, we must first `import` it.

```python
import numpy as np
```
In the previous line, we named the `numpy` *module* `np`; an abbreviation for easier referencing.


In `numpy`, an *array* is  a generic term for a multidimensional
set of numbers.
We use the `np.array()` function to define   `x` and `y`, which are one-dimensional arrays, i.e. vectors.

```python
x = np.array([3, 4, 5])
y = np.array([4, 9, 7])
```
Note that if you forgot to run the `import numpy as np` command earlier, then
you will encounter an error in calling the `np.array()` function in the previous line. 
 The syntax `np.array()` indicates that the function being called
is part of the `numpy` package, which we have abbreviated as `np`. 


Since `x` and `y` have been defined using `np.array()`, we get a sensible result when we add them together. Compare this to our results in the previous section,
 when we tried to add two lists without using `numpy`. 

```python
x + y
```
    
 



In `numpy`, matrices are typically represented as two-dimensional arrays, and vectors as one-dimensional arrays. {While it is also possible to create matrices using  `np.matrix()`, we will use `np.array()` throughout the labs in this book.}
We can create a two-dimensional array as follows. 

```python
x = np.array([[1, 2], [3, 4]])
x
```
    




The object `x` has several 
*attributes*, or associated objects. To access an attribute of `x`, we type `x.attribute`, where we replace `attribute`
with the name of the attribute. 
For instance, we can access the `ndim` attribute of  `x` as follows. 

```python
x.ndim
```

The output indicates that `x` is a two-dimensional array.  
Similarly, `x.dtype` is the *data type* attribute of the object `x`. This indicates that `x` is 
comprised of 64-bit integers:

```python
x.dtype
```
Why is `x` comprised of integers? This is because we created `x` by passing in exclusively integers to the `np.array()` function.
  If
we had passed in any decimals, then we would have obtained an array of
*floating point numbers* (i.e. real-valued numbers). 

```python
np.array([[1, 2], [3.0, 4]]).dtype

```


Typing `fun?` will cause `Python` to display 
documentation associated with the function `fun`, if it exists.
We can try this for `np.array()`. 

```python
np.array?

```
This documentation indicates that we could create a floating point array by passing a `dtype` argument into `np.array()`.

```python
np.array([[1, 2], [3, 4]], float).dtype

```


The array `x` is two-dimensional. We can find out the number of rows and columns by looking
at its `shape` attribute.

```python
x.shape

```


A *method* is a function that is associated with an
object. 
For instance, given an array `x`, the expression
`x.sum()` sums all of its elements, using the `sum()`
method for arrays. 
The call `x.sum()` automatically provides `x` as the
first argument to its `sum()` method.

```python
x = np.array([1, 2, 3, 4])
x.sum()
```
We could also sum the elements of `x` by passing in `x` as an argument to the `np.sum()` function. 

```python
x = np.array([1, 2, 3, 4])
np.sum(x)
```
 As another example, the
`reshape()` method returns a new array with the same elements as
`x`, but a different shape.
 We do this by passing in a `tuple` in our call to
 `reshape()`, in this case `(2, 3)`.  This tuple specifies that we would like to create a two-dimensional array with 
$2$ rows and $3$ columns. {Like lists, tuples represent a sequence of objects. Why do we need more than one way to create a sequence? There are a few differences between tuples and lists, but perhaps the most important is that elements of a tuple cannot be modified, whereas elements of a list can be.}
 
In what follows, the
`\n` character creates a *new line*.

```python
x = np.array([1, 2, 3, 4, 5, 6])
print("beginning x:\n", x)
x_reshape = x.reshape((2, 3))
print("reshaped x:\n", x_reshape)

```

The previous output reveals that `numpy` arrays are specified as a sequence
of *rows*. This is  called *row-major ordering*, as opposed to *column-major ordering*. 


`Python` (and hence `numpy`) uses 0-based
indexing. This means that to access the top left element of `x_reshape`, 
we type in `x_reshape[0,0]`.

```python
x_reshape[0, 0]
```
Similarly, `x_reshape[1,2]` yields the element in the second row and the third column 
of `x_reshape`. 

```python
x_reshape[1, 2]
```
Similarly, `x[2]` yields the
third entry of `x`. 

Now, let's modify the top left element of `x_reshape`.  To our surprise, we discover that the first element of `x` has been modified as well!



```python
print("x before we modify x_reshape:\n", x)
print("x_reshape before we modify x_reshape:\n", x_reshape)
x_reshape[0, 0] = 5
print("x_reshape after we modify its top left element:\n", x_reshape)
print("x after we modify top left element of x_reshape:\n", x)

```

Modifying `x_reshape` also modified `x` because the two objects occupy the same space in memory.
 

    


We just saw that we can modify an element of an array. Can we also modify a tuple? It turns out that we cannot --- and trying to do so introduces
an *exception*, or error.

```python
my_tuple = (3, 4, 5)
my_tuple[0] = 2

```


We now briefly mention some attributes of arrays that will come in handy. An array's `shape` attribute contains its dimension; this is always a tuple.
The  `ndim` attribute yields the number of dimensions, and `T` provides its transpose. 

```python
x_reshape.shape, x_reshape.ndim, x_reshape.T

```

Notice that the three individual outputs `(2,3)`, `2`, and `array([[5, 4],[2, 5], [3,6]])` are themselves output as a tuple. 
 
We will often want to apply functions to arrays. 
For instance, we can compute the
square root of the entries using the `np.sqrt()` function: 

```python
np.sqrt(x)

```

We can also square the elements:

```python
x**2

```

We can compute the square roots using the same notation, raising to the power of $1/2$ instead of 2.

```python
x**0.5

```


Throughout this book, we will often want to generate random data. 
The `np.random.normal()`  function generates a vector of random
normal variables. We can learn more about this function by looking at the help page, via a call to `np.random.normal?`.
The first line of the help page  reads `normal(loc=0.0, scale=1.0, size=None)`. 
 This  *signature* line tells us that the function's arguments are  `loc`, `scale`, and `size`. These are *keyword* arguments, which means that when they are passed into
 the function, they can be referred to by name (in any order). {`Python` also uses *positional* arguments. Positional arguments do not need to use a keyword. To see an example, type in `np.sum?`. We see that `a` is a positional argument, i.e. this function assumes that the first unnamed argument that it receives is the array to be summed. By contrast, `axis` and `dtype` are keyword arguments: the position in which these arguments are entered into `np.sum()` does not matter.}
 By default, this function will generate random normal variable(s) with mean (`loc`) $0$ and standard deviation (`scale`) $1$; furthermore, 
 a single random variable will be generated unless the argument to `size` is changed. 

We now generate 50 independent random variables from a $N(0,1)$ distribution. 

```python
x = np.random.normal(size=50)
x

```

We create an array `y` by adding an independent $N(50,1)$ random variable to each element of `x`.

```python
y = x + np.random.normal(loc=50, scale=1, size=50)
```
The `np.corrcoef()` function computes the correlation matrix between `x` and `y`. The off-diagonal elements give the 
correlation between `x` and `y`. 

```python
np.corrcoef(x, y)
```

If you're following along in your own `Jupyter` notebook, then you probably noticed that you got a different set of results when you ran the past few 
commands. In particular, 
 each
time we call `np.random.normal()`, we will get a different answer, as shown in the following example.

```python
print(np.random.normal(scale=5, size=2))
print(np.random.normal(scale=5, size=2))

```
    


In order to ensure that our code provides exactly the same results
each time it is run, we can set a *random seed* 
using the 
`np.random.default_rng()` function.
This function takes an arbitrary, user-specified integer argument. If we set a random seed before 
generating random data, then re-running our code will yield the same results. The
object `rng` has essentially all the random number generating methods found in `np.random`. Hence, to
generate normal data we use `rng.normal()`.

```python
rng = np.random.default_rng(1303)
print(rng.normal(scale=5, size=2))
rng2 = np.random.default_rng(1303)
print(rng2.normal(scale=5, size=2))
```

Throughout the labs in this book, we use `np.random.default_rng()`  whenever we
perform calculations involving random quantities within `numpy`.  In principle, this
should enable the reader to exactly reproduce the stated results. However, as new versions of `numpy` become available, it is possible
that some small discrepancies may occur between the output
in the labs and the output
from `numpy`.

The `np.mean()`,  `np.var()`, and `np.std()`  functions can be used
to compute the mean, variance, and standard deviation of arrays.  These functions are also
available as methods on the arrays.

```python
rng = np.random.default_rng(3)
y = rng.standard_normal(10)
np.mean(y), y.mean()
```
    


```python
np.var(y), y.var(), np.mean((y - y.mean())**2)
```


Notice that by default `np.var()` divides by the sample size $n$ rather
than $n-1$; see the `ddof` argument in `np.var?`.


```python
np.sqrt(np.var(y)), np.std(y)
```

The `np.mean()`,  `np.var()`, and `np.std()` functions can also be applied to the rows and columns of a matrix. 
To see this, we construct a $10 \times 3$ matrix of $N(0,1)$ random variables, and consider computing its row sums. 

```python
X = rng.standard_normal((10, 3))
X
```

Since arrays are row-major ordered, the first axis, i.e. `axis=0`, refers to its rows. We pass this argument into the `mean()` method for the object `X`. 

```python
X.mean(axis=0)
```

The following yields the same result.

```python
X.mean(0)
```
    


## Graphics
In `Python`, common practice is to use  the library
`matplotlib` for graphics.
However, since `Python` was not written with data analysis in mind,
  the notion of plotting is not intrinsic to the language. 
We will use the `subplots()` function
from `matplotlib.pyplot` to create a figure and the
axes onto which we plot our data.
For many more examples of how to make plots in `Python`,
readers are encouraged to visit [matplotlib.org/stable/gallery/](https://matplotlib.org/stable/gallery/index.html).

In `matplotlib`, a plot consists of a *figure* and one or more *axes*. You can think of the figure as the blank canvas upon which 
one or more plots will be displayed: it is the entire plotting window. 
The *axes* contain important information about each plot, such as its $x$- and $y$-axis labels,
title,  and more. (Note that in `matplotlib`, the word *axes* is not the plural of *axis*: a plot's *axes* contains much more information 
than just the $x$-axis and  the $y$-axis.)

We begin by importing the `subplots()` function
from `matplotlib`. We use this function
throughout when creating figures.
The function returns a tuple of length two: a figure
object as well as the relevant axes object. We will typically
pass `figsize` as a keyword argument.
Having created our axes, we attempt our first plot using its  `plot()` method.
To learn more about it, 
type `ax.plot?`.

```python
from matplotlib.pyplot import subplots

fig, ax = subplots(figsize=(8, 8))
x = rng.standard_normal(100)
y = rng.standard_normal(100)
ax.plot(x, y);

```

We pause here to note that we have *unpacked* the tuple of length two returned by `subplots()` into the two distinct
variables `fig` and `ax`. Unpacking
is typically preferred to the following equivalent but slightly more verbose code:

```python
output = subplots(figsize=(8, 8))
fig = output[0]
ax = output[1]
```

We see that our earlier cell produced a line plot, which is the default. To create a scatterplot, we provide an additional argument to `ax.plot()`, indicating that circles should be displayed.

```python
fig, ax = subplots(figsize=(8, 8))
ax.plot(x, y, "o");
```
Different values
of this additional argument can be used to produce different colored lines
as well as different linestyles. 



As an alternative, we could use the  `ax.scatter()` function to create a scatterplot.

```python
fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker="o");
```

Notice that in the code blocks above, we have ended
the last line with a semicolon. This prevents `ax.plot(x, y)` from printing
text  to the notebook. However, it does not prevent a plot from being produced. 
 If we omit the trailing semi-colon, then we obtain the following output:  

```python
fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker="o")

```
In what follows, we will use
 trailing semicolons whenever the text that would be output is not
germane to the discussion at hand.






To label our plot, we  make use of the `set_xlabel()`,  `set_ylabel()`, and  `set_title()` methods
of `ax`.
  

```python
fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker="o")
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y");
```

 Having access to the figure object `fig` itself means that we can go in and change some aspects and then redisplay it. Here, we change
  the size from `(8, 8)` to `(12, 3)`.


```python
fig.set_size_inches(12,3)
fig
```
 


Occasionally we will want to create several plots within a figure. This can be
achieved by passing additional arguments to `subplots()`. 
Below, we create a  $2 \times 3$ grid of plots
in a figure of size determined by the `figsize` argument. In such
situations, there is often a relationship between the axes in the plots. For example,
all plots may have a common $x$-axis. The `subplots()` function can automatically handle
this situation when passed the keyword argument `sharex=True`.
The `axes` object below is an array pointing to different plots in the figure. 

```python
fig, axes = subplots(nrows=2,
                     ncols=3,
                     figsize=(15, 5))
```
We now produce a scatter plot with `'o'` in the second column of the first row and
a scatter plot with `'+'` in the third column of the second row.

```python
axes[0,1].plot(x, y, "o")
axes[1,2].scatter(x, y, marker="+")
fig
```
Type  `subplots?` to learn more about 
`subplots()`. 





To save the output of `fig`, we call its `savefig()`
method. The argument `dpi` is the dots per inch, used
to determine how large the figure will be in pixels.

```python
fig.savefig("Figure.png", dpi=400)
fig.savefig("Figure.pdf", dpi=200);

```


We can continue to modify `fig` using step-by-step updates; for example, we can modify the range of the $x$-axis, re-save the figure, and even re-display it. 

```python
axes[0,1].set_xlim([-1,1])
fig.savefig("Figure_updated.jpg")
fig
```

We now create some more sophisticated plots. The 
`ax.contour()` method  produces a  *contour plot* 
in order to represent three-dimensional data, similar to a
topographical map.  It takes three arguments:

* A vector of `x` values (the first dimension),
* A vector of `y` values (the second dimension), and
* A matrix whose elements correspond to the `z` value (the third
dimension) for each pair of `(x,y)` coordinates.

To create `x` and `y`, we’ll use the command  `np.linspace(a, b, n)`, 
which returns a vector of `n` numbers starting at  `a` and  ending at `b`.

```python
fig, ax = subplots(figsize=(8, 8))
x = np.linspace(-np.pi, np.pi, 50)
y = x
f = np.multiply.outer(np.cos(y), 1 / (1 + x**2))
ax.contour(x, y, f);

```
We can increase the resolution by adding more levels to the image.

```python
fig, ax = subplots(figsize=(8, 8))
ax.contour(x, y, f, levels=45);
```
To fine-tune the output of the
`ax.contour()`  function, take a
look at the help file by typing `?plt.contour`.
 
The `ax.imshow()`  method is similar to 
`ax.contour()`, except that it produces a color-coded plot
whose colors depend on the `z` value. This is known as a
*heatmap*, and is sometimes used to plot temperature in
weather forecasts.

```python
fig, ax = subplots(figsize=(8, 8))
ax.imshow(f);

```


## Sequences and Slice Notation


As seen above, the
function `np.linspace()`  can be used to create a sequence
of numbers.

```python
seq1 = np.linspace(0, 10, 11)
seq1

```


The function `np.arange()`
 returns a sequence of numbers spaced out by `step`. If `step` is not specified, then a default value of $1$ is used. Let's create a sequence
 that starts at $0$ and ends at $10$.

```python
seq2 = np.arange(0, 10)
seq2

```

Why isn't $10$ output above? This has to do with *slice* notation in `Python`. 
Slice notation  
is used to index sequences such as lists, tuples and arrays.
Suppose we want to retrieve the fourth through sixth (inclusive) entries
of a string. We obtain a slice of the string using the indexing  notation  `[3:6]`.

```python
"hello world"[3:6]
```
In the code block above, the notation `3:6` is shorthand for  `slice(3,6)` when used inside
`[]`. 

```python
"hello world"[slice(3,6)]

```

You might have expected  `slice(3,6)` to output the fourth through seventh characters in the text string (recalling that  `Python` begins its indexing at zero),  but instead it output  the fourth through sixth. 
 This also explains why the earlier `np.arange(0, 10)` command output only the integers from $0$ to $9$. 
See the documentation `slice?` for useful options in creating slices. 

    



    


    

 

    

 

    


    



## Indexing Data
To begin, we  create a two-dimensional `numpy` array.

```python
A = np.array(np.arange(16)).reshape((4, 4))
A

```

Typing `A[1,2]` retrieves the element corresponding to the second row and third
column. (As usual, `Python` indexes from $0.$)

```python
A[1,2]

```

The first number after the open-bracket symbol `[`
 refers to the row, and the second number refers to the column. 

### Indexing Rows, Columns, and Submatrices
 To select multiple rows at a time, we can pass in a list
  specifying our selection. For instance, `[1,3]` will retrieve the second and fourth rows:

```python
A[[1,3]]

```

To select the first and third columns, we pass in  `[0,2]` as the second argument in the square brackets.
In this case we need to supply the first argument `:` 
which selects all rows.

```python
A[:,[0,2]]

```

Now, suppose that we want to select the submatrix made up of the second and fourth 
rows as well as the first and third columns. This is where
indexing gets slightly tricky. It is natural to try  to use lists to retrieve the rows and columns:

```python
A[[1,3],[0,2]]

```

 Oops --- what happened? We got a one-dimensional array of length two identical to

```python
np.array([A[1,0],A[3,2]])

```

 Similarly,  the following code fails to extract the submatrix comprised of the second and fourth rows and the first, third, and fourth columns:

```python
A[[1,3],[0,2,3]]

```

We can see what has gone wrong here. When supplied with two indexing lists, the `numpy` interpretation is that these provide pairs of $i,j$ indices for a series of entries. That is why the pair of lists must have the same length. However, that was not our intent, since we are looking for a submatrix.

One easy way to do this is as follows. We first create a submatrix by subsetting the rows of `A`, and then on the fly we make a further submatrix by subsetting its columns.


```python
A[[1,3]][:,[0,2]]

```
    


There are more efficient ways of achieving the same result.

The *convenience function* `np.ix_()` allows us  to extract a submatrix
using lists, by creating an intermediate *mesh* object.

```python
idx = np.ix_([1,3],[0,2,3])
A[idx]

```


Alternatively, we can subset matrices efficiently using slices.
  
The slice
`1:4:2` captures the second and fourth items of a sequence, while the slice `0:3:2` captures
the first and third items (the third element in a slice sequence is the step size).

```python
A[1:4:2,0:3:2]

```
    


Why are we able to retrieve a submatrix directly using slices but not using lists?
Its because they are different `Python` types, and
are treated differently by `numpy`.
Slices can be used to extract objects from arbitrary sequences, such as strings, lists, and tuples, while the use of lists for indexing is more limited.




    

 

    

 


### Boolean Indexing
In `numpy`, a *Boolean* is a type  that equals either   `True` or  `False` (also represented as $1$ and $0$, respectively).
The next line creates a vector of $0$'s, represented as Booleans, of length equal to the first dimension of `A`. 

```python
keep_rows = np.zeros(A.shape[0], bool)
keep_rows
```
We now set two of the elements to `True`. 

```python
keep_rows[[1,3]] = True
keep_rows

```

Note that the elements of `keep_rows`, when viewed as integers, are the same as the
values of `np.array([0,1,0,1])`. Below, we use  `==` to verify their equality. When
applied to two arrays, the `==`   operation is applied elementwise.

```python
np.all(keep_rows == np.array([0,1,0,1]))

```

(Here, the function `np.all()` has checked whether
all entries of an array are `True`. A similar function, `np.any()`, can be used to check whether any entries of an array are `True`.)


   However, even though `np.array([0,1,0,1])`  and `keep_rows` are equal according to `==`, they index different sets of rows!
The former retrieves the first, second, first, and second rows of `A`. 

```python
A[np.array([0,1,0,1])]

```

 By contrast, `keep_rows` retrieves only the second and fourth rows  of `A` --- i.e. the rows for which the Boolean equals `TRUE`. 

```python
A[keep_rows]

```

This example shows that Booleans and integers are treated differently by `numpy`.


We again make use of the `np.ix_()` function
 to create a mesh containing the second and fourth rows, and the first,  third, and fourth columns. This time, we apply the function to Booleans,
 rather than lists.

```python
keep_cols = np.zeros(A.shape[1], bool)
keep_cols[[0, 2, 3]] = True
idx_bool = np.ix_(keep_rows, keep_cols)
A[idx_bool]

```

We can also mix a list with an array of Booleans in the arguments to `np.ix_()`:

```python
idx_mixed = np.ix_([1,3], keep_cols)
A[idx_mixed]

```
    


For more details on indexing in `numpy`, readers are referred
to the `numpy` tutorial mentioned earlier.



## Loading Data

Data sets often contain different types of data, and may have names associated with the rows or columns. 
For these reasons, they typically are best accommodated using a
 *data frame*. 
 We can think of a data frame  as a sequence
of arrays of identical length; these are the columns. Entries in the
different arrays can be combined to form a row.
 The `pandas`
library can be used to create and work with data frame objects.


### Reading in a Data Set

The first step of most analyses involves importing a data set into
`Python`.  
 Before attempting to load
a data set, we must make sure that `Python` knows where to find the file containing it. 
If the
file is in the same location
as this notebook file, then we are all set. 
Otherwise, 
the command
`os.chdir()`  can be used to *change directory*. (You will need to call `import os` before calling `os.chdir()`.) 


We will begin by reading in `Auto.csv`, available on the book website. This is a comma-separated file, and can be read in using `pd.read_csv()`: 

```python
import pandas as pd

Auto = pd.read_csv("Auto.csv")
Auto

```

The book website also has a whitespace-delimited version of this data, called `Auto.data`. This can be read in as follows:

```python
Auto = pd.read_csv("Auto.data", delim_whitespace=True)

```
 Both `Auto.csv` and `Auto.data` are simply text
files. Before loading data into `Python`, it is a good idea to view it using
a text editor or other software, such as Microsoft Excel.




We now take a look at the column of `Auto` corresponding to the variable `horsepower`: 

```python
Auto["horsepower"]

```
We see that the `dtype` of this column is `object`. 
It turns out that all values of the `horsepower` column were interpreted as strings when reading
in the data. 
We can find out why by looking at the unique values.

```python
np.unique(Auto["horsepower"])

```
We see the culprit is the value `?`, which is being used to encode missing values.




To fix the problem, we must provide `pd.read_csv()` with an argument called `na_values`.
Now,  each instance of  `?` in the file is replaced with the
value `np.nan`, which means *not a number*:

```python
Auto = pd.read_csv("Auto.data",
                   na_values=["?"],
                   delim_whitespace=True)
Auto["horsepower"].sum()

```


The `Auto.shape`  attribute tells us that the data has 397
observations, or rows, and nine variables, or columns.

```python
Auto.shape

```

There are
various ways to deal with  missing data. 
In this case, since only five of the rows contain missing
observations,  we choose to use the `Auto.dropna()` method to simply remove these rows.

```python
Auto_new = Auto.dropna()
Auto_new.shape

```


### Basics of Selecting Rows and Columns
 
We can use `Auto.columns`  to check the variable names.

```python
Auto = Auto_new # overwrite the previous value
Auto.columns

```


Accessing the rows and columns of a data frame is similar, but not identical, to accessing the rows and columns of an array. 
Recall that the first argument to the `[]` method
is always applied to the rows of the array.  
Similarly, 
passing in a slice to the `[]` method creates a data frame whose *rows* are determined by the slice:

```python
Auto[:3]

```
Similarly, an array of Booleans can be used to subset the rows:

```python
idx_80 = Auto["year"] > 80
Auto[idx_80]

```
However, if we pass  in a list of strings to the `[]` method, then we obtain a data frame containing the corresponding set of *columns*. 

```python
Auto[["mpg", "horsepower"]]

```
Since we did not specify an *index* column when we loaded our data frame, the rows are labeled using integers
0 to 396.

```python
Auto.index

```
We can use the
`set_index()` method to re-name the rows using the contents of `Auto['name']`. 

```python
Auto_re = Auto.set_index("name")
Auto_re

```

```python
Auto_re.columns

```
We see that the column `'name'` is no longer there.
 
Now that the index has been set to `name`, we can  access rows of the data 
frame by `name` using the `{loc[]`} method of
`Auto`:

```python
rows = ["amc rebel sst", "ford torino"]
Auto_re.loc[rows]

```
As an alternative to using the index name, we could retrieve the 4th and 5th rows of `Auto` using the `{iloc[]`} method:

```python
Auto_re.iloc[[3,4]]

```
We can also use it to retrieve the 1st, 3rd and and 4th columns of `Auto_re`:

```python
Auto_re.iloc[:,[0,2,3]]

```
We can extract the 4th and 5th rows, as well as the 1st, 3rd and 4th columns, using
a single call to `iloc[]`:

```python
Auto_re.iloc[[3,4],[0,2,3]]

```
Index entries need not be unique: there are several cars  in the data frame named `ford galaxie 500`.

```python
Auto_re.loc["ford galaxie 500", ["mpg", "origin"]]

```
### More on Selecting Rows and Columns
Suppose now that we want to create a data frame consisting of the  `weight` and `origin`  of the subset of cars with 
`year` greater than 80 --- i.e. those built after 1980.
To do this, we first create a Boolean array that indexes the rows.
The `loc[]` method allows for Boolean entries as well as strings:

```python
idx_80 = Auto_re["year"] > 80
Auto_re.loc[idx_80, ["weight", "origin"]]

```


To do this more concisely, we can use an anonymous function called a `lambda`: 

```python
Auto_re.loc[lambda df: df["year"] > 80, ["weight", "origin"]]

```
The `lambda` call creates a function that takes a single
argument, here `df`, and returns `df['year']>80`.
Since it is created inside the `loc[]` method for the
dataframe `Auto_re`, that dataframe will be the argument supplied.
As another example of using a `lambda`, suppose that
we want all cars built after 1980 that achieve greater than 30 miles per gallon:

```python
Auto_re.loc[lambda df: (df["year"] > 80) & (df["mpg"] > 30),
            ["weight", "origin"],
           ]

```
The symbol `&` computes an element-wise *and* operation.
As another example, suppose that we want to retrieve all `Ford` and `Datsun`
cars with `displacement` less than 300. We check whether each `name` entry contains either the string `ford` or `datsun` using the  `str.contains()` method of the `index` attribute of 
of the dataframe:

```python
Auto_re.loc[lambda df: (df["displacement"] < 300)
                       & (df.index.str.contains("ford")
                       | df.index.str.contains("datsun")),
            ["weight", "origin"],
           ]

```
Here, the symbol `|` computes an element-wise *or* operation.
 
In summary, a powerful set of operations is available to index the rows and columns of data frames. For integer based queries, use the `iloc[]` method. For string and Boolean
selections, use the `loc[]` method. For functional queries that filter rows, use the `loc[]` method
with a function (typically a `lambda`) in the rows argument.

## For Loops
A `for` loop is a standard tool in many languages that
repeatedly evaluates some chunk of code while
varying different values inside the code.
For example, suppose we loop over elements of a list and compute their sum.

```python
total = 0
for value in [3,2,19]:
    total += value
print(f"Total is: {total}")

```
The indented code beneath the line with the `for` statement is run
for each value in the sequence
specified in the `for` statement. The loop ends either
when the cell ends or when code is indented at the same level
as the original `for` statement.
We see that the final line above which prints the total is executed
only once after the for loop has terminated. Loops
can be nested by additional indentation.

```python
total = 0
for value in [2,3,19]:
    for weight in [3, 2, 1]:
        total += value * weight
print(f"Total is: {total}")
```
Above, we summed over each combination of `value` and `weight`.
We also took advantage of the *increment* notation
in `Python`: the expression `a += b` is equivalent
to `a = a + b`. Besides
being a convenient notation, this can save time in computationally
heavy tasks in which the intermediate value of `a+b` need not
be explicitly created.

Perhaps a more
common task would be to sum over `(value, weight)` pairs. For instance,
to compute the average value of a random variable that takes on
possible values 2, 3 or 19 with probability 0.2, 0.3, 0.5 respectively
we would compute the weighted sum. Tasks such as this
can often be accomplished using the `zip()`  function that
loops over a sequence of tuples.

```python
total = 0
for value, weight in zip([2,3,19],
                         [0.2,0.3,0.5]):
    total += weight * value
print(f"Weighted average is: {total}")

```

### String Formatting
In the code chunk above we also printed a string
displaying the total. However, the object `total`
is an  integer and not a string.
Inserting the value of something into
a string is a common task, made
simple using
some of the powerful string formatting
tools in `Python`.
Many data cleaning tasks involve
manipulating and programmatically
producing strings.

For example we may want to loop over the columns of a data frame and
print the percent missing in each column.
Let’s create a data frame `D` with columns in which 20% of the entries are missing i.e. set
to `np.nan`.  We’ll create the
values in `D` from a normal distribution with mean 0 and variance 1 using `rng.standard_normal()`
and then overwrite some random entries using `rng.choice()`.

```python
rng = np.random.default_rng(1)
A = rng.standard_normal((127, 5))
M = rng.choice([0, np.nan], p=[0.8,0.2], size=A.shape)
A += M
D = pd.DataFrame(A, columns=["food",
                             "bar",
                             "pickle",
                             "snack",
                             "popcorn"])
D[:3]

```


```python
for col in D.columns:
    template = 'Column "{0}" has {1:.2%} missing values'
    print(template.format(col,
          np.isnan(D[col]).mean()))

```
We see that the `template.format()` method expects two arguments `{0}`
and `{1:.2%}`, and the latter includes some formatting
information. In particular, it specifies that the second argument should be expressed as a percent with two decimal digits.

The reference
[docs.python.org/3/library/string.html](https://docs.python.org/3/library/string.html)
includes many helpful and more complex examples.


## Additional Graphical and Numerical Summaries
We can use the `ax.plot()` or  `ax.scatter()`  functions to display the quantitative variables. However, simply typing the variable names will produce an error message,
because `Python` does not know to look in the  `Auto`  data set for those variables.

```python
fig, ax = subplots(figsize=(8, 8))
ax.plot(horsepower, mpg, "o");
```
We can address this by accessing the columns directly:

```python
fig, ax = subplots(figsize=(8, 8))
ax.plot(Auto["horsepower"], Auto["mpg"], "o");

```
Alternatively, we can use the `plot()` method with the call `Auto.plot()`.
Using this method,
the variables  can be accessed by name.
The plot methods of a data frame return a familiar object:
an axes. We can use it to update the plot as we did previously: 

```python
ax = Auto.plot.scatter("horsepower", "mpg")
ax.set_title("Horsepower vs. MPG");
```
If we want to save
the figure that contains a given axes, we can find the relevant figure
by accessing the `figure` attribute:

```python
fig = ax.figure
fig.savefig("horsepower_mpg.png");
```

We can further instruct the data frame to plot to a particular axes object. In this
case the corresponding `plot()` method will return the
modified axes we passed in as an argument. Note that
when we request a one-dimensional grid of plots, the object `axes` is similarly
one-dimensional. We place our scatter plot in the middle plot of a row of three plots
within a figure.

```python
fig, axes = subplots(ncols=3, figsize=(15, 5))
Auto.plot.scatter("horsepower", "mpg", ax=axes[1]);

```

Note also that the columns of a data frame can be accessed as attributes: try typing in `Auto.horsepower`. 


We now consider the `cylinders` variable. Typing in `Auto.cylinders.dtype` reveals that it is being treated as a quantitative variable. 
However, since there is only a small number of possible values for this variable, we may wish to treat it as 
 qualitative.  Below, we replace
the `cylinders` column with a categorical version of `Auto.cylinders`. The function `pd.Series()`  owes its name to the fact that `pandas` is often used in time series applications.

```python
Auto.cylinders = pd.Series(Auto.cylinders, dtype="category")
Auto.cylinders.dtype

```
 Now that `cylinders` is qualitative, we can display it using
 the `boxplot()` method.

```python
fig, ax = subplots(figsize=(8, 8))
Auto.boxplot("mpg", by="cylinders", ax=ax);

```

The `hist()`  method can be used to plot a *histogram*.

```python
fig, ax = subplots(figsize=(8, 8))
Auto.hist("mpg", ax=ax);

```
The color of the bars and the number of bins can be changed:

```python
fig, ax = subplots(figsize=(8, 8))
Auto.hist("mpg", color="red", bins=12, ax=ax);

```
 See `Auto.hist?` for more plotting
options.
 
We can use the `pd.plotting.scatter_matrix()`   function to create a *scatterplot matrix* to visualize all of the pairwise relationships between the columns in
a data frame.

```python
pd.plotting.scatter_matrix(Auto);

```
 We can also produce scatterplots
for a subset of the variables.

```python
pd.plotting.scatter_matrix(Auto[["mpg",
                                 "displacement",
                                 "weight"]]);

```
The `describe()`  method produces a numerical summary of each column in a data frame.

```python
Auto[["mpg", "weight"]].describe()

```
We can also produce a summary of just a single column.

```python
Auto["cylinders"].describe()
Auto["mpg"].describe()

```
To exit `Jupyter`,  select `File / Shut Down`.

 

