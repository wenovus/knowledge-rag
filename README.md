# Chat with your Knowledge Graph

## 🤖 Introduction

This repository features a Graph RAG (Retrieval-Augmented Generation) project that allows users to query complex Knowledge Graphs using a natural language, conversational interface.

Designed for real-world, textual graphs, this flexible Question Answering framework is highly versatile and applicable across diverse domains, including:

- Scene Graph Understanding
- Common Sense Reasoning
- Advanced Knowledge Graph Reasoning

## ⚡ Ablative Analysis

Our research systematically investigates the G-Retriever architecture [![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2402.07630). 
The core focus is an ablation analysis to rigorously quantify how modifications to the system's components impact the overall effectiveness and performance of the Knowledge Graph conversational system. 


## :rocket: Setup

### 🌱  Create an environment and install dependencies

Check out the **Managing dependencies** section of the **Contributing** guide to learn how to set up the environment and install dependencies.

### 🌱 Setting up env variables

You can use `.env` file for set the following enviromental variables. 

Create account in [Hugging Face](https://huggingface.co/settings/tokens) and get your token from [here](https://huggingface.co/settings/tokens).
```
export HF_TOKEN=<your_api_key>
```
Create account in  [W&B](https://wandb.ai/site) to track, visualize and manage the ablative analysis. To authenticate your machine with W&B, generate an API key from your user profile or at [here](https://wandb.ai/authorize).

```
export WANDB_API_KEY=<your_api_key>
```

## :pushpin: Version tools

Testing and implementation of this project were conducted on a system featuring **dual A100 80GB GPUs**. To ensure reproducible results for the ablative analysis, this **hardware configuration must be utilized**.

Key software versions used:

- Python: 3.12
- Cuda: 12.4 
- Torch: 2.6.0+cu124
- Torchvision: 0.21.0

Use the following commands to confirm your installed versions match the required dependencies:
```
nvidia-smi
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```


## ✅ How to replicate the ablative analysis

Each component of the ablative analysis is presented in its own dedicated section. For full reproducibility, we include a **brief description of the experiment's goal**, the **specific command needed for execution**, and a **W&B link to access the live metrics, logs, and definitive results**.

## 🔭 Data Preprocessing:

Prior to running the ablative analysis, the dataset requires preprocessing. This essential step includes textualizing the graph nodes and edges, embedding the questions, and then saving the resulting graphs in the PyTorch Geometric Data format. Finally, the dataset must be correctly split into train, validation, and test sets. Use the below commands to preprocess the datasets.

```
# ExplaGraphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

# SceneGraphs
python -m src.dataset.preprocess.scene_graphs

# WebQSP
python -m src.dataset.preprocess.webqsp

```


## 🔭 Varying Subgraph Retrieval Methods: 
We implemented two subgraph retrieval methods, K-hop and Personalized PageRank (PPR), and evaluated their performance using the SceneGraphs and WebQSP datasets.

```
# SceneGraphs
python -m src.dataset.scene_graphs --retrieval_method ppr
python -m src.dataset.scene_graphs --retrieval_method ppr

# WebQSP
python -m src.dataset.webqsp --retrieval_method k-hop
python -m src.dataset.webqsp --retrieval_method k-hop

```
## 🔭 Varying Subgraph Encoder Type:
We implemented two graph neural network (GNN) architectures for the G-retriever architecture, GraphSAGE and the Graph Isomorphism Network (GIN), and evaluated their performance using the ExplaGraphs and WebQSP benchmark datasets.

```
### GraphSAGE
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False  --gnn_model_name graphsage
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False  --gnn_model_name graphsage

### GIN
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False  --gnn_model_name gin
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False  --gnn_model_name gin
```
[GraphSAGE benchmarking for ExplaGraphs](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GraphSAGE-ExplaGraphs--VmlldzoxNTI0ODY2Nw?accessToken=t7s1tqhv0wru55i1jbp9dzatl2gdexjg80pupj0wa9qijvga7bwc3onmfsdeusba)

[GraphSAGE benchmarking for WebQSP](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GraphSAGE-WebQSP--VmlldzoxNTI0ODk2OA?accessToken=rl84ivm79vt5cufdob8ajvne7ecrjpwlxardi18u8uzqc1ugo75uoccguuntbef9)

[GIN benchmarking for ExplaGraphs](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GIN-ExplaGraphs--VmlldzoxNTI0ODY5MQ?accessToken=9i50yn52010qz3zwurlj9wjuhenucgy3gm1x1479xrrqpataq96rqbjtc2punoab)

[GIN benchmarking for WebQSP](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GIN-WebQSP--VmlldzoxNTI1OTYwNg?accessToken=shf3cmrptkodzfsfskwpuv7au6vfz4n35n7e3l4w59y03982yu4xmfgqujgx05uq)


## 🔭 Varying LLM Models:
We benchmarked a set of large language models (LLMs) with similar feature dimensions to Llama-2–7b-hf, including Gemma-7b, Mistral-7B-v0.1, Qwen-8B, and Deepseek-Coder-6.7b-base. Evaluation was performed using two representative datasets, ExplaGraphs and WebQSP, along with their respective LLM-instructed variants.

```

```
## 🔭 Prompt Tuning:
We implemented a specific system prompt template and measured its effect on performance when applied to the WebQSP knowledge graph question answering dataset.

```

```
## 🔭 New Graph RAG Model Architecture:
We propose a novel model architecture that integrates prompt tuning and G-Retriever with LoRA-based Large Language Model (LLM) fine-tuning. We then measure its performance using the WebQSP knowledge graph question answering dataset.

```

```

























## Data Preprocessing
```
# expla_graphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

# scene_graphs, might take
python -m src.dataset.preprocess.scene_graphs
python -m src.dataset.scene_graphs

# webqsp
python -m src.dataset.preprocess.webqsp
python -m src.dataset.webqsp
```

## Training
Replace path to the llm checkpoints in the `src/model/__init__.py`, then run

### 1) Inference-Only LLM
```
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat

python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat


```
### 2) Frozen LLM + Prompt Tuning
```
# prompt tuning
python train.py --dataset scene_graphs_baseline --model_name pt_llm

python train.py --dataset expla_graphs --model_name pt_llm


# G-Retriever
python train.py --dataset scene_graphs --model_name graph_llm
```

### 3) Tuned LLM
```
# finetune LLM with LoRA
python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False

# G-Retriever with LoRA
python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False
```

## Reproducibility
Use `run.sh` to run the codes and reproduce the published results in the main table.


## Samples retrieval graphs 

python -m src.dataset.preprocess.scene_graphs_sample
python -m src.dataset.scene_graphs_sample

python -m src.dataset.preprocess.webqsp_sample
python -m src.dataset.webqsp_sample

python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs