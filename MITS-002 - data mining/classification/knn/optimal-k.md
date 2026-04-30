### Methods for Determining Optimal K in K-Means Clustering

--- 
#### Common Rule of Thumb

**Square Root of N:** A quick heuristic is to set *K* where is the total number of training samples.
$$ k = \sqrt{\frac{N}{2}} $$ 
$$ k = \sqrt{N} $$
- **Lall and Sharma (1996)** - *A Nearest Neighbor Function Fitting Algorithm for Hydrologic Forecasting*
- **Devroye, Györfi, and Lugosi (1996)** - *A Probabilistic Theory of Pattern Recognition*
- **Fix and Hodges (1951)** -  While they introduced the KNN concept itself, they focused on the general rule that 
 must be a function of to ensure the estimated density converges to the true density

**Odd Numbers:** It is generally recommended to choose an odd value for *k* in binary classification to prevent ties in voting. eg. 
*[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]*
- **Richard O. Duda, Peter E. Hart, and David G. Stork** - [Pattern Classification](https://www.wiley.com/en-us/Pattern+Classification%2C+2nd+Edition-p-9780471056690)


**Cross Validation:** The standard empirical approach often recommended to find the *k* that yields the lowest error rate, commonly implemented via GridSearch eg. scikit-learn in python
- **M. Stone (1974)** - [Cross-Validatory Choice and Assessment of Statistical Predictions](https://academic.oup.com/jrsssb/article/36/2/111/7027414)  in the Journal of the Royal Statistical Society\

**Grid Search:** - to be continue

--- 
**Elbow Method**
[Robert L. Thorndike (1953)](https://taylorandfrancis.com/knowledge/Engineering_and_technology/Computer_science/Elbow_method/) - The Elbow method is a heuristic approach that calculates the Total Within-Cluster Sum of Squares (WCSS) for various values of *K*. As *K* 
increases, WCSS decreases. The "elbow" is the point where the rate of decrease changes sharply, indicating diminishing returns. [wikipedia](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
- other contributors:
    - **Modern Analysis:** The method is frequently discussed in modern data mining literature as a standard, albeit sometimes subjective, technique (e.g., [Syakur et al.](https://iopscience.iop.org/article/10.1088/1757-899X/336/1/012017), [Cui](https://www.clausiuspress.com/article/592.html)).
    - **Critique and Alternatives:** [Samuele Mazzanti](https://medium.com/data-science/are-you-still-using-the-elbow-method-5d271b3063bd) has criticized the method, citing it as less reliable than techniques like the Calinski-Harabasz index or BIC.
    - **Refinements:** Researchers like [Petras and Masyn (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092417/) have suggested improvements like the "difference method" to reduce the subjectivity of picking the "elbow" point.

![Alt Text](https://media.licdn.com/dms/image/v2/D4D12AQF-yYtbzPvNFg/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1682277078758?e=2147483647&v=beta&t=BWIfjY3GR82JytDIAg9d8F8AYnzZdOD6ZmFOOYSmkE8)

---

#### Conclusion
| Value | Model Complexity | Bias vs. Variance | Error Risk |
| -------- | -------- | -------- | -------- |
| **Small**   | High complexity   | Low Bias / High Variance   | Overfitting |
| **Optimal**   | Balanced | Sweet Spot | 	Minimum Validation Error |
| **Large** | Low complexity  | High Bias / Low Variance | Underfitting |