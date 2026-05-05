# Evaluating Alignment of Behavioral Dispositions in LLMs

As LLMs integrate into our daily lives, understanding their behavior becomes
essential. In this work, we focus on behavioral dispositions-the underlying
tendencies that shape responses in social contexts-and introduce a framework to
study how closely the dispositions expressed by LLMs align with those of humans.
Our approach is grounded in established psychological questionnaires but adapts
them for LLMs by transforming human self-report statements into Situational
Judgment Tests (SJTs). These SJTs assess behavior by eliciting natural
recommendations in realistic user-assistant scenarios. We generate 2,500 SJTs,
each validated by three human annotators, and collect preferred actions from 10
annotators per SJT, from a large pool of 550 participants. In a comprehensive
study involving 25 LLMs, we find that models often do not reflect the
distribution of human preferences: (1) in scenarios with low human consensus,
LLMs consistently exhibit overconfidence in a single response; (2) when human
consensus is high, smaller models deviate significantly, and even some frontier
models do not reflect the consensus in 15-20% of cases; (3) traits can exhibit
cross-LLM patterns, e.g., LLMs may encourage emotion expression in contexts
where human consensus favors composure. Lastly, mapping psychometric statements
directly to behavioral scenarios presents a unique opportunity to evaluate the
predictive validity of self-reports, revealing considerable gaps between LLMs'
stated values and their revealed behavior.

<div align="center">
  <img src="figure_1.png" alt="Bird's-eye view of our framework." width="800"/>
</div>

This repository contains the code for reproducing the results in the paper.

## Overview

The evaluation pipeline tests whether LLMs exhibit behavioral dispositions
(e.g., Empathy, Impulsiveness, Assertiveness, Emotion Regulation) that align
with human judgments. It uses a **Situational Judgment Test (SJT)** methodology:

1. **Dataset**: A set of scenarios where a user presents a dilemma, along with
   two possible actions (agree/oppose) and human annotations.
2. **Action Generation**: An evaluated LLM is prompted with each scenario
   multiple times to estimate its behavioral tendency.
3. **Judge Grading**: A judge LLM classifies each free-text response as
   agreeing with one of the two actions.
4. **Alignment Analysis**: The model's action distribution is compared against
   human annotator consensus to measure directional alignment.

## Dataset

The SJT dataset is available on Kaggle:
https://www.kaggle.com/datasets/conferencerelease/alignment-of-behavioral-dispositions-in-llms/data

## Quick Start

Open `behavioral_dispositions_eval.ipynb` in Google Colab and follow the
step-by-step instructions. You will need a
[Google AI Studio API key](https://aistudio.google.com/apikey).

The notebook runs the full pipeline end-to-end:

1. Downloads the SJT dataset from Kaggle.
2. Runs a Gemini model on each scenario (with multiple replications) to
   generate free-text action recommendations.
3. Uses a judge Gemini model to classify each response as AGREE or OPPOSE.
4. Computes the **Directional Alignment** metric — measuring how well the
   model's behavioral tendencies match human consensus — broken down by
   personality trait and consensus strength (reproducing Figure 4 of the paper).
5. Saves all intermediate and final results to CSV.

## Directory Structure

```
behavioral_dispositions/
├── README.md                              # This file
├── requirements.txt                       # pip dependencies
├── behavioral_dispositions_eval.ipynb     # End-to-end Colab notebook
└── src/
    ├── __init__.py                        # Package init
    ├── utils.py                           # Shared utilities
    ├── gemini_runner.py                   # Gemini AI Studio API runner
    ├── action_generation.py               # Stage 1: Run model on SJT scenarios
    ├── judge_grading.py                   # Stage 2: Grade responses with judge
    └── alignment_analysis.py              # Stage 3: Compute alignment metrics
```
