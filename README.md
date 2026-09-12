# Hallucination Detector

A browser-based demo and research project for checking whether a generated response stays grounded in the source material it is meant to reflect. The project brings together a lightweight claim-level verification workflow, a trained model, and a visual interface that makes the detection process easy to explore.

## Project overview

This repository contains a claim verification demo built around the idea of testing factual consistency between a source document and a response. Instead of simply asking whether a sentence sounds plausible, the approach breaks the response into claims and compares each claim against the source text using lexical grounding signals such as similarity, novelty, numeric consistency, and entity overlap.

The goal is to flag unsupported or weakly grounded statements while keeping the interface simple and interactive enough for demonstration use.

## What is included

- An interactive browser demo in claim_verifier_demo.html
- A project folder with example inputs, exported model data, and demo artifacts
- Training and evaluation scripts for the claim verification pipeline
- Result files with metrics and example predictions
- A report-generation workflow for summarizing model behavior and outcomes

## Repository structure

- claim_verifier_demo.html — main interactive demo
- hallucination_detector_project/demo/ — example data and exported model payload
- hallucination_detector_project/src/ — Python code for dataset processing, feature engineering, training, and evaluation
- hallucination_detector_project/results/ — saved metrics and prediction outputs
- hallucination_detector_project/writeup/ — report generation assets

## Method

The verification pipeline works at the claim level. Each sentence in a response is treated as a separate claim and evaluated against the relevant source text. The system compares the claim to the source using a set of matching and mismatch features, including:

- similarity between the claim and relevant source passages
- novel-word and unsupported-content signals
- number and capitalization mismatches
- position and phrase-length indicators
- task-specific adjustments for summary, QA, and structured-data settings

These features are fed to a trained gradient-boosted classifier, and the model output is used to mark a claim as supported or likely unsupported.

## Demo focus

The front-end demo is designed to showcase the idea in a clean, presentation-friendly layout. It provides a live verification experience where a user can paste source text and response text, review the claim-by-claim verdicts, and inspect the model signal behind each classification.

The app is intentionally lightweight and static, which makes it easy to open locally and present without needing a backend service.

## Results and artifacts

The project includes evaluation outputs and example data that show how the model behaves on sample claim sets. These files are stored in the results directory and help document both the strengths and the limitations of the approach.

## Summary

This project sits at the intersection of NLP evaluation, fact verification, and interactive demonstration. It is meant to be readable, approachable, and useful as a prototype for understanding how grounding checks can be built and visualized in a practical way.
