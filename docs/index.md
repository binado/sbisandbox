# SBI Sandbox

## About Simulation-based Inference

The problem of inference is ubiquitous in modern science. In the past decades, many statistical tools have seen their use consolidated across different scientific fields, either through frequentist or Bayesian approaches.

In parallel, the design of more powerful and more efficient computing hardware has allowed for the design of high-fidelity simulations of physical systems with increasing complexity, allowing for the generation of synthetic data from them. However, performing inference from these simulators still remains challenging. In these contexts, the likelihood function is not explicitly calculated, and is instead implicitly defined by the data-generating process implemented by the simulator. The problem of performing inference with these systems has thus been named *likelihood-free inference* or *simulation-based inference* (hereafter SBI).

In the past few years, the development of more sophisticated Machine Learning techniques, in particular deep neural networks, and the production of specialized hardware for training has given new momentum to the field of SBI. For a high-level overview of the impact of these trends on the emergence of new methods, we refer the reader to [this review paper by (Cranmer et al, 2019).](https://arxiv.org/abs/1911.01429)

The number of scientific publications employing the SBI toolbox in their methodology has been rampant in the last couple of years. We highlight https://simulation-based-inference.org/, an automated aggregator of scientific articles related to the subject and spanning many different fields, such as statistics, economics, neuroscience, astrophysics and cosmology, epidemiology and ecology, and so on. Similarly, the Github repository https://github.com/smsharma/awesome-neural-sbi contains a curated list of publications, tutorials and software packages related to SBI.

## References

[1]: Cranmer, Kyle, Johann Brehmer, and Gilles Louppe. "The frontier of simulation-based inference." Proceedings of the National Academy of Sciences 117.48 (2020): 30055-30062.

[2]: Tejero-Cantero, Alvaro, et al. "SBI--A toolkit for simulation-based inference." arXiv preprint arXiv:2007.09114 (2020).

[3]: Kobyzev, Ivan, Simon JD Prince, and Marcus A. Brubaker. "Normalizing flows: An introduction and review of current methods." IEEE transactions on pattern analysis and machine intelligence 43.11 (2020): 3964-3979.

[4]: Lueckmann, Jan-Matthis, et al. "Benchmarking simulation-based inference." International conference on artificial intelligence and statistics. PMLR, 2021.
