+++
title = "Gradient Boosting"
authors = ["Alex Dillhoff"]
date = 2023-07-17T00:00:00-05:00
tags = ["machine learning"]
draft = false
lastmod = 2024-07-07
sections = "Machine Learning"
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Notes from (<a href="#citeproc_bib_item_1">Friedman 2001</a>)](#notes-from)

</div>
<!--endtoc-->



## Notes from (<a href="#citeproc_bib_item_1">Friedman 2001</a>) {#notes-from}

-   Many machine learning methods are parameterized functions that are optimized using some numerical optimization techniques, notably steepest-descent.
-   Initial learner is a stump, subsequent learners are trees with depth as some power of 2 (commonly).
-   **Numerical optimization in function space**
    \\[
      g\_m(\mathbf{x}) = E\_y\Big[\frac{\partial L(y, F(\mathbf{x}))}{\partial F(\mathbf{x})}|\mathbf{x}\Big]\_{F(\mathbf{x})=F\_{m-1}(\mathbf{x})}
      \\]
    The optimal step size found by solving

    \\[
      \rho\_m = \mathop{\arg \min}\_{\rho} E\_{y,\mathbf{x}}L(y,F\_{m-1}(\mathbf{x})-\rho g\_m(\mathbf{x}))
      \\]
    Then the function \\(m\\) is updated:
    \\[
      f\_m(\mathbf{x}) = -\rho\_m g\_m(\mathbf{x})
      \\]

    Walking through it...

    1.  Make an initial guess with \\(f\_0(\mathbf{x})\\)

    2.  Evaluate \\(L(y, f\_0(\mathbf{x}))\\)

    3.  Improve model by boosting \\(f\_1(\mathbf{x}) = -\rho\_1 g\_1(\mathbf{x})\\), where \\[ g\_1(\mathbf{x}) = \frac{\partial L(y, f\_0(\mathbf{x}))}{\partial f\_0(\mathbf{x})}. \\]
        This implies that \\(f\_1\\) is predicting the gradient of the previous function.

-   If the model is nonparametric, the expected value of the function conditioned on the input cannot be estimated accurately because we cannot sample the entire distribution of \\(\mathbf{x}\\). The author's note that "...even if it could, one would like to estimate \\(F^\*(\mathbf{x})\\) at \\(\mathbf{x}\\) values other than the training sample points."
    -   Smoothness is imposed by approximating the function with a parametric model. I think this means that the distribution is approximated as well.

        \begin{equation}
        (\beta\_m, \mathbf{a}\_m) = \mathop{\arg \min}\_{\beta, \mathbf{a}}\sum\_{i=1}^N L(y\_i, F\_{m-1}(\mathbf{x}\_i) + \beta h(\mathbf{x}\_i; \mathbf{a}))
        \end{equation}

    -   What if a solution to the above equation is difficult to obtain? Instead, view \\(\beta\_m h(\mathbf{x};\mathbf{a}\_m)\\) as the best greedy step toward \\(F^\*(\mathbf{x})\\), under the constraint that the step direction, in this case \\(h(\mathbf{x};\mathbf{a}\_m)\\), is a member of the class of functions \\(h(\mathbf{x};\mathbf{a})\\). The negative gradient can be evaluated at each data point:
        \\[
            -g\_m(\mathbf{x}\_i) = -\frac{\partial L(y\_i, F\_{m-1}(\mathbf{x}\_i))}{\partial F\_{m-1}(\mathbf{x}\_i)}.
            \\]
    -   This gradient is evaluated at every data point. However, we cannot generalize to new values not in our dataset. The proposed solution comes via \\(\mathbf{h}\_m = \\{h(\mathbf{x}\_i;\mathbf{a}\_m)\\}\_{1}^N\\) "most parallel to" \\(-\mathbf{g}\_m \in \mathbb{R}^N\\).
    -   As long as we can compute a derivative for the original loss function, our subsequent boosting problems are solved via least-squared error:
        \\[
            \mathbf{a}\_m = \mathop{\arg \min}\_{\mathbf{a}, \beta} \sum\_{i=1}^N \Big(-g\_m(\mathbf{x}\_i)-\beta h(\mathbf{x}\_i;\mathbf{a})\Big)^2
            \\]

        {{< figure src="/ox-hugo/2023-07-18_19-43-31_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Original generic algorithm from (<a href=\"#citeproc_bib_item_1\">Friedman 2001</a>)." >}}

        Check out a basic implementation in Python [here](<https://github.com/ajdillhoff/CSE6363/blob/main/boosting/intro_to_gradient_boosting.ipynb>).

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Friedman, Jerome H. 2001. “Greedy Function Approximation: A Gradient Boosting Machine.” <i>The Annals of Statistics</i> 29 (5): 1189–1232. <a href="https://www.jstor.org/stable/2699986">https://www.jstor.org/stable/2699986</a>.</div>
</div>
