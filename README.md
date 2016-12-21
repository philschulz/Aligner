# Aligner
This software contains several Bayesian word alignment models, based on IBM1 and IBM2 as well as the HMM aligner. It contains
the improvements to these models reported in [Schulz et al. (2016)](https://www.aclweb.org/anthology/P/P16/P16-2028.pdf) and [Schulz and Aziz (2016)](https://aclweb.org/anthology/C/C16/C16-1296.pdf).

You can directly download the jar file and run it using 
```bash
java -jar Aligner.jar
```
This will display the help menu with all the options. The most important of these is the model. Another important option is the number of iterations. In the paper we used 1000 iterations (from which we took 40 samples) but this is probably too much. About 250 iterations (with a sample lag of 5 or no sample lag at all) should do just fine. 

For the collocation-based models we recommend using hyperparameter inference. It hardly takes any time but avoids having to set the collocation hyperparameters by hand. For the translation and lexical hyperparameters we recommend a value of 0.0001.
