# Third-party notices

The code in this repository is under the MIT licence in [LICENSE](LICENSE). The processed data in `data/processed/` and the per-message predictions in `results/predictions/` are derived from two public datasets, redistributed under their own licences.

## BongoScam / Swahili SMS Detection Dataset

Henry Dioniz (2024), [Kaggle](https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset), released under the MIT licence. Changes made here: exact duplicates removed; phone numbers, money amounts and links replaced by `<PHONE>`, `<AMOUNT>` and `<URL>`; messages that became identical after masking removed; obfuscated and edited copies generated for evaluation.

```
MIT License

Copyright (c) 2024 Henry Dioniz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SMS Fraud Classification dataset for Chichewa

Taylor, A., & Robert, A. (2025), [Zenodo, doi:10.5281/zenodo.14607454](https://doi.org/10.5281/zenodo.14607454), released under the [Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/). Only the `D_CHI` and `telcoSMS_CHI` sheets are used. Changes made here: phone numbers, money amounts and links masked as above; messages that became identical after masking removed; near-duplicate templates marked.

## Models

The fine-tuned model published at [supserrr/wdyri-afroxlmr](https://huggingface.co/supserrr/wdyri-afroxlmr) is derived from [Davlan/afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base) (Alabi et al., 2022), itself adapted from [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base) (Conneau et al., 2020), both MIT-licensed. The BiLSTM uses the [fastText Swahili vectors](https://fasttext.cc/docs/en/crawl-vectors.html) (Grave et al., 2018), CC BY-SA 3.0, which are downloaded by `src/download.py` and not redistributed.
