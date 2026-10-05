# Research questions and contributions

## Problem

Mobile-money scams arrive as SMS that pose as relatives, agents, landlords or employers and ask the victim to send money; in Tanzania a common hook is *"ni tumie kwa namba hii"* (send it to me on this number). Published Swahili detectors report 98.7% (BongoScam) to 99.86% (Mambina et al., 2022) accuracy. WDYRI reproduces that result and then asks what prior work skipped:

1. **RQ1, leakage.** How much of the score survives when near-duplicate scam templates cannot appear in both train and test?
2. **RQ2, robustness.** How far does each model fall under rule-based obfuscation (lookalike letters, split words, English code-switching), and do simple defences recover it?
3. **RQ3, transfer.** Can a model trained only on Swahili catch real Chichewa fraud SMS from Malawi, and does Africa-centric pretraining (AfroXLMR) beat plain XLM-R there?

Answering them raised a fourth question, which became the project's main contribution: **what are these models actually reading?**

**Users:** mobile-money customers checking a suspicious SMS, and the telcos or fintechs that could run the check for them.

## Is the task already solved?

The brief's "no established system" rule is written for machine translation (Swahili has established MT, so it would be barred there). For text classification the matching question is whether *this task in this language* is already solved by an established system. We checked every Swahili scam-SMS detector we could find:

| Existing work | What exists | Why it does not settle the task |
| --- | --- | --- |
| Mambina et al. (2022) | 99.86% accuracy, random forest | Data not public; scams under 1% of messages; no robustness or transfer test |
| BongoScam (Dioniz, 2024) | public data + Naive Bayes app, 98.7% | One random split; we show its scores rest partly on a phone-number shortcut (E3b, E10) |
| Njame et al. (2026) | F1 about 0.998 | In-distribution only |
| Chiuseni et al. (2026) | obfuscation study | LLM-written attacks; defences proposed, not tested; English-Swahili only |
| Taylor & Robert (2025a) | Chichewa fraud SMS, about 97% | Classical models; no cross-language transfer |

No public, robustly evaluated Swahili scam classifier exists, and none for Chichewa beyond classical in-language models. Swahili is chosen because it has public, real-world scam SMS and the most published detectors to test against; Chichewa extends the question to a lower-resource neighbour that XLM-R never saw in pretraining.

## What is new

1. **A shortcut, found and measured.** In BongoScam, 86% of scams and 0% of genuine texts contain a phone number or link; the fixed rule "has a number" scores F1 0.92 on the whole dataset. Minimal-pair stress tests (E10) show the cost: appending a phone number to a genuine message makes fine-tuned AfroXLMR call it a scam **96%** of the time, and deleting the number from a scam makes it miss **20%**.
2. **A targeted fix.** Number-balanced counterfactual training (E11) edits one token in 42% of training texts so that a number appears about equally often in both classes. AfroXLMR's stress-test false alarms fall from 96% to **0%** and its misses from 20% to **2%**; its test PR-AUC rises from 0.958 to 0.985 (the F1 gain, 0.78 to 0.82, is not significant, p = 0.06). The fix has costs: for AfroXLMR, lower Chichewa precision; for the char LR and BiLSTM, weaker resistance to lookalike letters; and the number-balanced AfroXLMR flags disguised genuine texts more often.
3. **Leakage-safe evaluation.** Template-disjoint splits, ten re-drawn splits per protocol, a per-template view, seed-pooled bootstrap intervals and paired significance tests, instead of one random split.
4. **Reproducible attacks with controls and a second attacker.** Three rule-based attacks at three intensities, two tested defences, control sets that disguise *genuine* texts the same way, and a second attacker model. The controls show that the transformers' resistance to disguise is largely suspicion of odd-looking text: fully disguised genuine texts are flagged 54-93% of the time.
5. **A complementary ensemble, checked out of sample and chosen on validation.** A character n-gram model and AfroXLMR fail on different messages; flagging when either flags (both members number-balanced, chosen on validation stress tests) scores test F1 0.994 and keeps 0.95-0.99 under lookalike and split-word attacks from either attacker (+0.22 to +0.23 over the char LR, p < 0.001). On two fresh splits it beats AfroXLMR by 0.20-0.22 (p < 0.001) and matches the char LR (+0.02-0.03, not significant). It is the deployed app.
6. **Swahili to Chichewa transfer.** Zero-shot transfer is weak: trained models reach fraud F1 0.46-0.74, and a 9-word length rule scores 0.72. AfroXLMR ranks Chichewa fraud better than XLM-R (PR-AUC +0.15, p < 0.001) and adapts faster from 50 examples (F1 0.87 vs 0.78, p < 0.001), but a char LR without pretraining adapts just as fast (0.88).

## Findings in one minute

* **E1:** the published 98.7% reproduces exactly (98.68% accuracy).
* **RQ1:** over 10 re-drawn splits, keeping templates out of the test costs character n-gram models 2.6-5.7 F1 points (char LR 0.998 → 0.972, char NB 0.995 → 0.938) and word models about 1 point or less; all four gaps are significant (Mann-Whitney p ≤ 0.02). The bigger effect is *which* scripts land in the test: one unseen landlord-impersonation script (31 of 81 test scams; no number, link or amount) is caught by word NB (31 of 31) and word LR (29 of 31) but not by the transformers (0-13%), whose test F1 is 0.76-0.78 while their per-template F1 is 0.98-0.99.
* **RQ2:** lookalike letters cut word-level LR recall from 0.96 to 0.30; normalisation restores it to 0.95 (+0.52 F1, p < 0.001) at no cost on clean text, but only partly repairs split words. The transformers lose nothing under disguise, but the controls show why: at full intensity they also flag disguised *genuine* texts (AfroXLMR 54-60%, XLM-R 93%), and the scams AfroXLMR catches only once disguised are mostly the landlord script it misses when clean. The char LR never flags disguised genuine texts but loses 21-24% of its F1. Results are the same with a second attacker. Code-switching is the attack normalisation cannot undo; adversarial training cuts the char LR's loss there from 19% to 4%.
* **RQ3:** zero-shot fraud F1 on 733 Chichewa SMS is 0.46-0.74 for trained models, against 0.68 for the phone rule and 0.72 for the length rule. AfroXLMR's ranking advantage over XLM-R is significant, but a random-initialised BiLSTM ranks as well (PR-AUC 0.80), so African pretraining is not the only route. Twenty Chichewa examples lift char LR from 0.65 to 0.82 on the same messages.
* **The shortcut:** removing it changes what transfers. Number-balanced AfroXLMR keeps its Chichewa F1 (0.72) but loses precision (-0.06, p < 0.001); number-balanced char LR loses most of its transfer (0.65 → 0.15): for the classical models, "contains a number" *was* the transfer.

## Related work

Swahili smishing detection has so far been evaluated on random splits with classical models (Mambina et al., 2022; Njame et al., 2026; BongoScam). Near-duplicate overlap between train and test is a known source of inflated NLP scores (Elangovan et al., 2021); our template-disjoint split applies that lesson to scam scripts. Models that exploit dataset artifacts instead of the intended signal are well documented (Gururangan et al., 2018; McCoy et al., 2019; Geirhos et al., 2020); our E10 minimal pairs follow the behavioural-testing approach of CheckList (Ribeiro et al., 2020), and E11 adapts counterfactually augmented data (Kaushik et al., 2020) to a measured shortcut. Visual and invisible-character attacks are known to break NLP systems (Eger et al., 2019; Boucher et al., 2022); Chiuseni et al. (2026) showed classical smishing detectors collapse under LLM-written obfuscation, and we test the defences they proposed with reproducible, rule-based attacks. Zero-shot cross-lingual transfer is known to be weak for distant or under-resourced targets and to improve quickly with a few target examples (Lauscher et al., 2020); AfroXLMR (Alabi et al., 2022) adapts XLM-R (Conneau et al., 2020) to African languages including Chichewa, which Taylor & Robert (2025a) studied only with classical in-language models.
