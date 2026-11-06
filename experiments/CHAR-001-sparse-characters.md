# CHAR-001: sparse character profiles

Status: predeclared; no results yet.

Model remains the pinned LiquidAI LFM3-1.7B. Try a new extraction method, preserving previous failed persona sweeps: teacher-force eight hand-authored neutral/villain/hamster response triples under the same neutral system and paired user prompts. Average each channel's gated MLP input over non-special response tokens, with equal weight per response. This captures text/style/content associations, not a proven character representation. Text lengths differ and lexical content is a confound.

For each style/layer, rank channels by absolute paired mean difference divided by neutral response RMS, requiring sign agreement on at least six of eight pairs. Select sixteen before looking at evaluation text. Normalize their standardized differences to unit L2, then multiply each channel by its neutral RMS to form a sparse native-activation profile. Retain all 24-layer rankings and feature tensors locally. Test layers 4, 10, 15 and 22, spanning sliding attention, full attention and convolution blocks.

Calibrate profile amplitude to 5% and 15% mean relative L2 change at post_ffn_norm on eight neutral prompt prefills, excluding special tokens and position zero. Cap the multiplier at 32; report any unreachable target. Record actual BF16 effects on four calibration prefills. Controls use sixteen random channels per layer, excluding both style selections, Gaussian signed weights and the same achieved-strength calibration. Include a reversed villain profile at layer 10 / 15%. No channel/persona is selected using evaluation outputs.

Six disjoint evaluation prompts: three playful scenarios and three strict checks. Compare identical neutral baselines, ordinary persona prompting, internal profiles at both strengths, sparse random controls at both strengths, and the reversed profile. Planned: 6 baselines + 12 prompted characters + 96 style-profile replies + 48 random replies + 6 reversed replies + 2 hook checks = **170 generations**. Greedy, cap 128. All outcomes remain recorded, including no effect and damage. Fixed seed controls profile/control/trial selection, not independent generation replicates.

Record matched baseline token changes, actual prefill strength, EOS, repetition, narrow exact checks, transparent character vocabulary hits, timing and raw text. Vocabulary is a browsing aid, never a personality/quality score. Describe tone, content, structure and damage separately. An internal character claim requires coherent transfer across fresh contexts, stronger evidence than random controls, task preservation and later replication. Ordinary prompting is a practical comparison, not proof the internal edit works.

Artifacts: source/config hashes, teacher-forced features, ranked channel maps, sparse recipes, float32 calibration and BF16 checks, shuffled trial schedule, raw replies, direct-comparison workbench and reviewed report/figures. No training, weight rewrite, admin or paid compute.
