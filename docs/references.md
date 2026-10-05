# References

Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In *Proceedings of the 29th International Conference on Computational Linguistics* (pp. 4336-4349). https://aclanthology.org/2022.coling-1.382

Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics, 5*, 135-146. https://doi.org/10.1162/tacl_a_00051

Boucher, N., Shumailov, I., Anderson, R., & Papernot, N. (2022). Bad characters: Imperceptible NLP attacks. In *2022 IEEE Symposium on Security and Privacy* (pp. 1987-2004). https://doi.org/10.1109/SP46214.2022.9833641

Chiuseni, D., Bahizire, A., Hama, S., & Ndibwile, J. D. (2026). *Adversarial robustness in smishing detection: A comparative analysis of adversarial fragility in classical vs. transformer-based detection systems* (arXiv:2608.12889). arXiv. https://arxiv.org/abs/2608.12889

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440-8451). https://doi.org/10.18653/v1/2020.acl-main.747

Dioniz, H. (2024). *Swahili SMS detection dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset

Eger, S., Şahin, G. G., Rücklé, A., Lee, J.-U., Schulz, C., Mesgar, M., Swarnkar, K., Simpson, E., & Gurevych, I. (2019). Text processing like humans do: Visually attacking and shielding NLP systems. In *Proceedings of NAACL-HLT 2019* (pp. 1634-1647). https://aclanthology.org/N19-1165

Elangovan, A., He, J., & Verspoor, K. (2021). Memorization vs. generalization: Quantifying data leakage in NLP performance evaluation. In *Proceedings of the 16th Conference of the European Chapter of the ACL* (pp. 1325-1335). https://aclanthology.org/2021.eacl-main.113

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*, 665-673. https://doi.org/10.1038/s42256-020-00257-z

Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018). Learning word vectors for 157 languages. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*. https://aclanthology.org/L18-1550

Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S. R., & Smith, N. A. (2018). Annotation artifacts in natural language inference data. In *Proceedings of NAACL-HLT 2018, Volume 2 (Short Papers)* (pp. 107-112). https://aclanthology.org/N18-2017

Kaushik, D., Hovy, E., & Lipton, Z. C. (2020). Learning the difference that makes a difference with counterfactually-augmented data. In *International Conference on Learning Representations (ICLR 2020)*. https://arxiv.org/abs/1909.12434

Lauscher, A., Ravishankar, V., Vulić, I., & Glavaš, G. (2020). From zero to hero: On the limitations of zero-shot language transfer with multilingual transformers. In *Proceedings of EMNLP 2020* (pp. 4483-4499). https://aclanthology.org/2020.emnlp-main.363

Mambina, I. S., Ndibwile, J. D., & Michael, K. F. (2022). Classifying Swahili smishing attacks for mobile money users: A machine-learning approach. *IEEE Access, 10*, 83061-83074. https://doi.org/10.1109/ACCESS.2022.3196464

McCoy, R. T., Pavlick, E., & Linzen, T. (2019). Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 3428-3448). https://aclanthology.org/P19-1334

Njame, R. A., Sanga, G., & Tende, I. (2026). A machine-learning model for phishing detection in Swahili messages: A case of Tanzania. *East African Journal of Information Technology, 9*(2), 97-114. https://doi.org/10.37284/eajit.9.2.5655

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 4902-4912). https://aclanthology.org/2020.acl-main.442

Taylor, A., & Robert, A. (2025a). Using machine learning to detect fraudulent SMSs in Chichewa. In *Integrating AI in Science, Management, and Technology (AISMT 2025)*, Communications in Computer and Information Science, vol. 2699. Springer. https://doi.org/10.1007/978-3-032-08260-2_12

Taylor, A., & Robert, A. (2025b). *SMS fraud classification dataset for Chichewa* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14607454

Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of EMNLP 2020: System Demonstrations* (pp. 38-45). https://doi.org/10.18653/v1/2020.emnlp-demos.6

## Acknowledgements

Datasets by Henry Dioniz (BongoScam) and Taylor & Robert (Chichewa). Pretrained models: XLM-R (Meta AI), AfroXLMR (Alabi et al.), fastText Swahili vectors (Grave et al.). Libraries: scikit-learn, PyTorch, Hugging Face Transformers, gensim, Gradio. Existing resources are cited above; the data pipeline, attacks, stress tests, counterfactual training, ensemble, evaluation and app in this repository were implemented for this project.
