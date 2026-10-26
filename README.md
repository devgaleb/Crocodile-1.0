<div align="center">

<img src="assets/crocodile-banner.png" alt="Crocodile 2.2 Logo" width="100%">

# 🐊 Crocodile 2.2

**SMARTER • FASTER • STRONGER**

*AI Video Generation • Research • Creative Tools*

</div>

---

### Open video-generation research code • Text-to-video • Image-to-video

**A community-maintained project presentation for the Crocodile 2.2
codebase.**

```{=html}
<p>
```
`<img alt="Version" src="https://img.shields.io/badge/version-2.2-38d430?style=for-the-badge">`{=html}
`<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">`{=html}
`<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">`{=html}
`<img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-green?style=for-the-badge">`{=html}
```{=html}
</p>
```
[Features](#features) · [Repository setup](#repository-setup) ·
[Models](#models-and-checkpoints) · [Quick start](#quick-start) ·
[Hardware](#hardware-and-performance) ·
[Credits](#credits-and-attribution)
:::

------------------------------------------------------------------------

## Overview

Crocodile 2.2 is a research-oriented video-generation codebase for
generating video from text prompts, animating still images, and
exploring related multimodal generation workflows. The repository
contains inference scripts, model components, utility modules, and
distributed-execution support.

This GitHub repository is maintained as a community-facing project. The
**Crocodile** name, logo, and README presentation do not imply that the
repository maintainer trained the underlying models or owns the original
research.

## Features

-   **Text-to-video (T2V):** generate a video from a written prompt.
-   **Image-to-video (I2V):** use a still image as the visual starting
    point for a generated clip.
-   **Text-and-image-to-video (TI2V):** combine a prompt and an image in
    supported workflows.
-   **Speech-driven video workflows:** optional components are provided
    for speech-to-video tasks.
-   **Character animation tools:** the codebase includes
    animation-related modules.
-   **High-resolution workflows:** supported output size depends on the
    selected model checkpoint and available hardware.
-   **Distributed inference utilities:** includes code for multi-GPU
    execution.
-   **Configurable inference:** options include model offloading and
    prompt extension in supported scripts.

Actual output quality, supported resolutions, speed, and memory use
depend on the checkpoint, runtime environment, settings, and hardware.
Features are not guaranteed to work until tested in your environment.

## Repository structure

``` text
Crocodile-1.0/
├── assets/                 # Project images and README branding
├── crocodile/              # Model code, pipelines, and utilities
│   ├── distributed/        # Distributed inference helpers
│   ├── modules/            # Model and neural-network components
│   └── utils/              # Supporting utilities
├── tests/                  # Test scripts
├── generate.py             # Main generation entry point
├── requirements.txt        # Core Python dependencies
├── requirements_s2v.txt    # Optional speech-to-video dependencies
├── requirements_animate.txt# Optional animation dependencies
├── INSTALL.md              # Additional installation notes
├── LICENSE.txt             # License text
└── README.md
```

The exact files in your branch may differ as the project evolves.

## Repository setup

### 1. Clone the repository

``` bash
git clone https://github.com/devgaleb/Crocodile-1.0.git
cd Crocodile-1.0
```

### 2. Create an isolated Python environment

Python 3.10 or newer is expected by the project metadata. A compatible
PyTorch build and NVIDIA GPU/CUDA setup are recommended for inference.

``` bash
python -m venv .venv
```

Activate it:

**Linux**

``` bash
source .venv/bin/activate
```

**Windows PowerShell**

``` powershell
.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

``` bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

Install a PyTorch build that matches your operating system and CUDA
setup, following the official PyTorch installation selector first. Then
install the project dependencies:

``` bash
pip install -r requirements.txt
```

Optional speech-to-video dependencies:

``` bash
pip install -r requirements_s2v.txt
```

Optional animation dependencies:

``` bash
pip install -r requirements_animate.txt
```

Some packages, particularly GPU-specific attention libraries, may
require a compatible Linux, CUDA, compiler, and PyTorch combination. If
installation fails, consult `INSTALL.md` and the package's official
instructions.

## Models and checkpoints

The code repository and model weights are separate. Download the
checkpoint required by your chosen task and make note of its exact local
directory. Use only checkpoints you are authorized to access and follow
their individual licenses and usage conditions.

The generation examples below use a text-to-video checkpoint directory
named `./Crocodile-2.2-T2V-A14B`. Replace that path if your checkpoint
is stored elsewhere.

For large checkpoints, ensure you have enough free disk space and
bandwidth. Do not commit model weights, API keys, or other large/private
files to Git unless the repository is explicitly configured for them.

## Quick start

### Text-to-video

After installing dependencies and downloading a compatible checkpoint:

``` bash
python generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Crocodile-2.2-T2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --prompt "A tiny glass greenhouse in a rainforest at sunrise, cinematic light, slow camera movement"
```

This is an example command, not a guarantee that it will run on every
GPU. Large checkpoints can require substantial GPU memory. If you
encounter an out-of-memory error, check the model documentation and
available options for offloading, resolution, and dtype.

### Multi-GPU example

For a machine with eight suitably configured GPUs, the codebase includes
distributed inference support. One example for a compatible environment
is:

``` bash
torchrun --nproc_per_node=8 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Crocodile-2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --prompt "A quiet futuristic city reflected in rain-soaked streets at blue hour"
```

The number of processes and parallelism settings must match your
hardware and environment. Do not run this command unchanged on a
single-GPU or CPU-only device.

### Prompt extension

Some workflows support prompt extension through a remote API or a local
language model. Remote services may require an API key and may incur
charges. Store credentials in environment variables or a secrets
manager; never place them directly in source files or commit them to
Git.

For the full list of flags and task names available in your checked-out
version, run:

``` bash
python generate.py --help
```

## Hardware and performance

Video generation is computationally intensive. The memory requirement
depends on the checkpoint size, output resolution, frame count,
precision, and inference options.

-   A compatible NVIDIA GPU is recommended for practical inference.
-   CUDA, PyTorch, and GPU library versions must be mutually compatible.
-   Multi-GPU commands require the corresponding number of available and
    correctly configured GPUs.
-   CPU offloading may reduce VRAM pressure but can increase runtime and
    system RAM use.
-   Mobile devices and Termux are useful for repository management, but
    this codebase's full model inference stack should not be assumed to
    run directly on Android.

No benchmark results are claimed by this README. Please publish measured
hardware, settings, and timings when sharing performance reports.

## Testing and development

The repository includes test-related files. Review their contents before
executing scripts, and run tests only in an environment with the
necessary dependencies and checkpoints.

``` bash
bash tests/test.sh
```

Formatting and development tools may be installed using the optional
development dependencies described in `pyproject.toml` or `INSTALL.md`.

## Troubleshooting

```{=html}
<details>
```
```{=html}
<summary>
```
`<strong>`{=html}Dependency or GPU-library installation
fails`</strong>`{=html}
```{=html}
</summary>
```
Confirm your Python, PyTorch, CUDA, compiler, and GPU-library versions
are compatible. Install the matching PyTorch build first, then follow
the relevant dependency's official installation instructions. Some GPU
libraries are not available on every operating system or architecture.
```{=html}
</details>
```
```{=html}
<details>
```
```{=html}
<summary>
```
`<strong>`{=html}CUDA out-of-memory error`</strong>`{=html}
```{=html}
</summary>
```
Try a smaller supported resolution, model offloading, or other
memory-saving options documented by the selected task. Close other GPU
processes and confirm that the checkpoint matches the inference script.
```{=html}
</details>
```
```{=html}
<details>
```
```{=html}
<summary>
```
`<strong>`{=html}Checkpoint or configuration not found`</strong>`{=html}
```{=html}
</summary>
```
Check that `--ckpt_dir` points to the downloaded checkpoint directory,
that all required files are present, and that you selected a task
compatible with that checkpoint.
```{=html}
</details>
```
```{=html}
<details>
```
```{=html}
<summary>
```
`<strong>`{=html}Prompt extension fails`</strong>`{=html}
```{=html}
</summary>
```
Check whether your selected extension method requires network access, an
API key, or a separately installed local model. Keep secrets out of your
command history and repository.
```{=html}
</details>
```
## Responsible use

Use generated media responsibly. Respect privacy, consent, copyright,
and the terms attached to datasets and checkpoints. Clearly disclose
synthetic media where appropriate, and do not use the software to
impersonate people or mislead others.

## Contributing

Issues and pull requests are welcome. When reporting a problem, include:

-   Operating system and Python version
-   PyTorch and CUDA versions, if applicable
-   GPU model and available VRAM
-   The task and relevant command (remove all secrets first)
-   A concise error log and steps to reproduce

Keep pull requests focused, document behavior changes, and avoid
committing checkpoints, generated outputs, caches, credentials, or
environment files.

## Credits and attribution

Crocodile 2.2 is a community-maintained presentation of an existing
research codebase. Please retain all original copyright notices, license
files, citations, and required attributions included with the source.
Renaming repository branding does not transfer ownership of the
underlying research, code, weights, or trademarks.

Read [`LICENSE.txt`](LICENSE.txt) and any model-specific license before
redistributing or using checkpoints. If you publish research or
derivative work, cite the original research and follow the citation
instructions supplied by its authors.

## License

This repository includes an Apache License 2.0 text in `LICENSE.txt`.
Individual checkpoints, third-party dependencies, datasets, and linked
services may have separate terms. Review those terms independently
before use or redistribution.

------------------------------------------------------------------------

::: {align="center"}
**CROCODILE 2.2**\
*Explore. Create. Iterate.*

Maintained by [@devgaleb](https://github.com/devgaleb)
:::
