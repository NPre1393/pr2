# PR2
Uni Vienna Praktikum 2
## Granger Causality for Time Series Analysis

Algorithm test pipeline for finding granger-causal relations

Requires Python >= 3.6, matlab for python / octave installed

---------------------------------------------------------------------------------------------------------------
examples
------------------------------------------------
contains tests for autoencoder models to develope a granger causal graph algorithm
that will be integrated into the pipeline

---------------------------------------------------------------------------------------------------------------
testing_pipeline - gc_testing
-----------------------------
* apps - contains two main parts of the testing pipeline

* test_algs - contains algorithms that can be run by the software

	* [dca_bi (bicgl)](https://www.researchgate.net/publication/338943678_Bi-directional_Causal_Graph_Learning_through_Weight-Sharing_and_Low-Rank_Neural_Network)
	* [gcf](https://www.researchgate.net/publication/328845941_Discovering_Granger-Causal_Features_from_Deep_Learning_Networks_31st_Australasian_Joint_Conference_Wellington_New_Zealand_December_11-14_2018_Proceedings)
	* [neunetnue](https://www.researchgate.net/publication/279632576_Neural_Networks_with_Non-Uniform_Embedding_and_Explicit_Validation_Phase_to_Assess_Granger_Causality)
	* [neural_gc](https://www.researchgate.net/publication/323257187_Neural_Granger_Causality_for_Nonlinear_Time_Series)

* tests - contains various tests for each algorithm and data generator/algorithm loader components

* test_nb - python notebooks that run similar tests, with output shown

* requirements.txt - python packages needed to run the pipeline

