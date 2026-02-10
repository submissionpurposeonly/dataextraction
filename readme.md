SHIELDA: A General-Purpose Exception-Handling Framework for LLM Agents
This repository contains the official implementation, datasets, and expanded runtime logs for SHIELDA, a framework-agnostic middleware designed to diagnose and resolve execution and reasoning-level exceptions in LLM agentic workflows.

🌟 Overview
SHIELDA introduces a layered defense mechanism (Classifier → Handler → Escalator) to break LLM agents out of "doom loops" caused by tool failures, goal drifts, or logical paradoxes. Unlike traditional error handling, SHIELDA operates as a non-intrusive interceptor, making it compatible with diverse agent frameworks like LangChain and Hugging Face.

Repository Structure
- shielda_core/: Contains the core logic of the SHIELDA framework, including the implementation of the Classifier, Handler, and Escalator components. It also includes the optimized Prompts used to drive the exception-handling pipeline.

- agent_source_code/: Provides the source code for the three agent architectures used in our evaluation: the standard ReAct agent, the LangChain ReAct agent, and the Hugging Face agent. This section demonstrates SHIELDA's framework-agnostic integration.

- case_studies/: A collection of raw, turn-by-turn execution logs (JSON format) for the qualitative analyses presented in Appendices B through H. Each log contains the original failure, the SHIELDA intervention, and the successful recovery trajectory.

- data/: Includes the evaluation assets for our quantitative experiments. This directory contains the Ground Truth files for the GAIA, WebShop, and ALFWorld benchmarks, as well as the original error logs that served as the starting point for our recovery tests.

-data-extraction.xlsx: Includes the detailed data used for the exception taxonomy and the paper.