# PAE_katana

PersonalRecAgent (PAE) - LLM-powered Personalized Recommendation System for WWW'25 AgentSocietyChallenge.

Katana cluster deployment package.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/SongyuQi-Francisco/PAE_katana.git
cd PAE_katana

# 2. Setup environment (one-time)
chmod +x setup_environment.sh
./setup_environment.sh

# 3. Download dataset (see Dataset section below)

# 4. Run full pipeline
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

## Project Structure

```
PAE_katana/
├── websocietysimulator/     # Core framework (from AgentSocietyChallenge)
│   ├── agent/               # Base agent classes
│   ├── llm/                 # LLM client implementations
│   ├── tools/               # InteractionTool, EvaluationTool
│   └── simulator.py         # Main evaluation harness
├── src/                     # Our agent implementation
│   ├── single_stage_rec_agent.py   # Main recommendation agent
│   ├── personal_rec_router.py      # Dynamic skill routing
│   └── skills_db.json              # Skill library
├── scripts/                 # Utility scripts
│   ├── extract_cognitive_profiles.py
│   ├── evolution_engine.py
│   └── ...
├── data/
│   ├── problem_splits/classic_3000/  # Dev train/val tasks (included)
│   └── user_profiles.json            # Pre-computed profiles (included)
├── tasks/track2/            # Official track2 evaluation tasks (included)
├── dataset/                 # Large dataset files (NOT included, download separately)
│   ├── user.json            # ~1.0 GB
│   ├── item.json            # ~1.3 GB
│   └── review.json          # ~4.0 GB
├── results/                 # Output directory
├── run_experiment.py        # Single experiment runner
├── run_full_pipeline.sh     # Full V0→V1 pipeline
├── setup_environment.sh     # Environment setup
└── requirements.txt
```

## Dataset Download

The dataset (~6.3 GB total) is NOT included in the repository. Download from:

**Option 1: Direct transfer from local machine**
```bash
# On your local machine, run:
scp -r /path/to/dataset user@katana:/path/to/PAE_katana/dataset/
```

**Option 2: From original source**
The dataset is from AgentSocietyChallenge. Contact organizers or download from:
- [AgentSocietyChallenge GitHub](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)

Required files in `dataset/`:
- `user.json` (~1.0 GB) - User profiles
- `item.json` (~1.3 GB) - Item metadata (Amazon, Yelp, Goodreads)
- `review.json` (~4.0 GB) - User reviews

## Environment Setup

### Prerequisites
- Python >= 3.10
- Conda (recommended)
- CUDA (optional, for GPU acceleration)

### Automatic Setup
```bash
./setup_environment.sh
```

### Manual Setup
```bash
conda create -n pae_katana python=3.10 -y
conda activate pae_katana
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

### Full Pipeline (Recommended)
```bash
# Runs: V0 baseline -> Skill Evolution -> V1 evolved -> Track2 evaluation
./run_full_pipeline.sh
```

### Custom Configuration
```bash
# Run specific domains only
DOMAINS="amazon" ./run_full_pipeline.sh

# Adjust parallelism
MAX_WORKERS=4 ./run_full_pipeline.sh

# Skip track2 final evaluation
RUN_TRACK2=0 ./run_full_pipeline.sh
```

### Single Experiment
```bash
conda activate pae_katana
# API key is loaded automatically from .env file

# V0 baseline (seed skills only)
ENABLE_EVOLUTION="false" python run_experiment.py \
    --task_dir data/problem_splits/classic_3000/train/amazon/tasks \
    --groundtruth_dir data/problem_splits/classic_3000/train/amazon/groundtruth \
    --task_set amazon \
    --version v0_test \
    --max_workers 8

# V1 evolved (with evolved skills)
ENABLE_EVOLUTION="true" python run_experiment.py \
    --task_dir data/problem_splits/classic_3000/train/amazon/tasks \
    --groundtruth_dir data/problem_splits/classic_3000/train/amazon/groundtruth \
    --task_set amazon \
    --version v1_test \
    --max_workers 8
```

## Katana PBS Job Script

For Katana cluster submission:

```bash
#!/bin/bash
#PBS -N pae_experiment
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pae_katana

./run_full_pipeline.sh > experiment.log 2>&1
```

Submit with: `qsub job_script.pbs`

## API Configuration

Create a `.env` file in the project root with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your API key:
# OPENAI_API_KEY=sk-your-key-here
```

Or set it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Output

Results are saved to `results/` directory:
```
results/
├── proposal_v0/          # V0 baseline results
│   ├── amazon_v0_devtrain_metrics.json
│   ├── yelp_v0_devtrain_metrics.json
│   └── goodreads_v0_devtrain_metrics.json
├── proposal_v1/          # V1 evolved results
│   └── ...
└── proposal_final/       # Track2 final evaluation
    └── ...
```

## Metrics

- **Primary**: HR@1, HR@3, HR@5 (Hit Rate at K)
- **Secondary**: NDCG@1, NDCG@3, NDCG@5

## Troubleshooting

### Rate Limit Errors (429)
OpenAI API has daily request limits. Solutions:
1. Reduce `MAX_WORKERS` to slow down requests
2. Wait for quota reset (usually 24h)
3. Use a different API key

### Missing Dataset
Ensure `dataset/` contains all three JSON files:
```bash
ls -la dataset/
# Should show user.json, item.json, review.json
```

### Memory Issues
Reduce batch size or workers:
```bash
MAX_WORKERS=2 ./run_full_pipeline.sh
```
