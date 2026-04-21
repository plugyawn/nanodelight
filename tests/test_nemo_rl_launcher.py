import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nemo_grpo_launcher_dry_run_is_test_only(tmp_path):
    nemo_rl_root = tmp_path / "NeMo-RL"
    nemo_rl_root.mkdir()
    run_dir = tmp_path / "dry-run"

    env = os.environ.copy()
    env.update(
        {
            "NEMO_RL_ROOT": str(nemo_rl_root),
            "APPLY_PATCHES": "0",
            "DRY_RUN": "1",
            "RUN_DIR": str(run_dir),
        }
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "runs/nemo_grpo_gsm8k_full.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    command_text = (run_dir / "command.txt").read_text()
    assert "data.train.split_validation_size=0.0" in command_text
    assert "data.train.split_validation_size=0.01" not in command_text
    assert "data.train.split_validation_size=0.05" not in command_text
    assert "++data.validation.dataset_name=gsm8k" in command_text
    assert "++data.validation.split=test" in command_text
    assert "DRY_RUN=1, command written" in result.stdout


def test_kondo_launcher_gsm8k_bringup_profile(tmp_path):
    nemo_rl_root = tmp_path / "NeMo-RL"
    nemo_rl_root.mkdir()
    run_dir = tmp_path / "kondo-gsm8k"

    env = os.environ.copy()
    env.update(
        {
            "NEMO_RL_ROOT": str(nemo_rl_root),
            "APPLY_PATCHES": "0",
            "DRY_RUN": "1",
            "RUN_DIR": str(run_dir),
            "EVAL_PROFILE": "gsm8k",
        }
    )

    subprocess.run(
        ["bash", str(REPO_ROOT / "runs/nemo_grpo_kondo_eval.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    command_text = (run_dir / "command.txt").read_text()
    assert "data.train.dataset_name=gsm8k" in command_text
    assert "++data.validation.dataset_name=gsm8k" in command_text
    assert "++data.validation.split=test" in command_text
    assert "++data.validation.dataset_name=AIME2024" not in command_text
    assert "grpo.kondo.enabled=1" in command_text
    assert "policy.model_name=Qwen/Qwen2.5-1.5B-Instruct" in command_text
    assert "grpo.max_num_steps=8" in command_text
    assert "grpo.num_prompts_per_step=2" in command_text
    assert "grpo.num_generations_per_prompt=4" in command_text
    assert "policy.max_total_sequence_length=768" in command_text
    assert "policy.generation.max_new_tokens=320" in command_text
    assert "policy.generation.vllm_cfg.max_model_len=768" in command_text
    assert "grpo.kondo.mode=response_dense_rows_v2" in command_text
    assert "grpo.kondo.target_backward_token_fraction=0.7" in command_text
    assert "grpo.kondo.recall_floor=1.0" in command_text
    assert "grpo.kondo.block_size=" not in command_text


def test_kondo_launcher_aime2024_qualification_profile(tmp_path):
    nemo_rl_root = tmp_path / "NeMo-RL"
    nemo_rl_root.mkdir()
    run_dir = tmp_path / "kondo-aime"

    env = os.environ.copy()
    env.update(
        {
            "NEMO_RL_ROOT": str(nemo_rl_root),
            "APPLY_PATCHES": "0",
            "DRY_RUN": "1",
            "RUN_DIR": str(run_dir),
            "EVAL_PROFILE": "aime2024",
        }
    )

    subprocess.run(
        ["bash", str(REPO_ROOT / "runs/nemo_grpo_kondo_eval.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    command_text = (run_dir / "command.txt").read_text()
    assert "data.train.dataset_name=OpenMathInstruct-2" in command_text
    assert "++data.validation.dataset_name=AIME2024" in command_text
    assert "++data.validation.repeat=16" in command_text
    assert "++data.validation.dataset_name=gsm8k" not in command_text
    assert "data.train.split_validation_size=0.05" not in command_text
