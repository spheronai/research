# Prior-art check: camera move reader

Week 1, checked 24 Nov 2026. Research only. Every claim below links to the page that backs it. Anything marked **unverified** could not be confirmed from a primary page during this check.

## Verdict: GO, with a changed angle

Nobody has shipped all three of these together: open weights, an importable per-frame camera path with focal length, and named moves, measured on a benchmark. The closest work, CameraVLM, does two of the three on paper but has not released weights, exports only NumPy arrays, and its poses are well behind VGGT. So the gap is real. But the plain "SfM gets geometry, VLMs get names" pitch from CameraBench is already being answered by at least four 2026 papers. We need a sharper claim.

**Changed angle we recommend:**

1. **Do not train a new pose model.** Sit on top of a commercially usable geometry backbone (VGGT-1B-Commercial, MapAnything-apache, or the Apache-licensed Depth Anything 3 sizes) and put our effort into the parts nobody owns:
   - **Zoom and focal length over time.** This is the clearest measured hole. Geometry methods score badly on zoom in CameraBench.
   - **Temporal segmentation with speed and easing** ("eases out").
   - **Clean export** to Blender, USD, FBX, Nuke and the RealEstate10K-style text file that Wan Fun Camera and CameraCtrl read.
2. **Make the "re-shoot test" the headline benchmark.** No paper found here scores a predicted camera by re-rendering the scene with it.
3. **Report the standard numbers alongside it.** That means CameraBench (eval only) and, where licences allow, CamChoreo and ACaM-Bench, so the move-naming half is comparable to existing work.
4. **Licensing is a selling point.** Most strong geometry baselines are non-commercial. An Apache-licensed end-to-end tool would be unusual.

**Risk to watch:** CameraVLM could publish weights at any time. If it does, it becomes the main baseline for both halves, and we compare on export quality, zoom and the re-shoot test.

### Three most important findings

1. **CameraVLM is the closest prior art, and it is not released.** It was posted to arXiv on 12 Oct 2026 as [a 9B VLM with a camera head and flow tokens](https://arxiv.org/abs/2610.17242).
   - [Its README](https://github.com/nicolascc1/CameraVLM) shows an API that returns `camera_to_world`, `intrinsics` and point tracks in a `geometry.npz` file, plus a JSON with `primary_motion`, `direction` and `confidence`.
   - The same README marks "Model weights", "Inference code" and "Evaluation code" as not done, with a "coming soon" badge.
   - [In its own pose table](https://arxiv.org/html/2610.17242), its ScanNet ATE is 0.124 against VGGT's 0.035, and its TUM ATE is 0.049 against VGGT's 0.009.
2. **Geometry methods fail on zoom in CameraBench, and fine-tuned VLMs do much better there.** On the [CameraBench leaderboard](https://sy77777en.github.io/CameraBench/leaderboard/table.html), average precision (AP) for zoom-in / zoom-out is:

   | Model | Zoom-in AP | Zoom-out AP |
   | --- | --- | --- |
   | MegaSaM | 11.1 | 10.2 |
   | CUT3R | 15.9 | 21.3 |
   | Qwen2.5-VL-72B fine-tuned on CameraBench (SFT) | 57.1 | 78.5 |

   On average AP the SFT model (62.8) still beats MegaSaM (58.2). Recovering focal length over time is the most defensible technical contribution.
3. **The licence landscape is mostly non-commercial.**
   - **Non-commercial:** CUT3R, MASt3R, Spann3R and MonST3R are CC BY-NC-SA 4.0. Pi3/Pi3X weights are CC BY-NC 4.0. VGGT-1B is CC BY-NC 4.0. VGGT-Omega is under the FAIR Noncommercial Research License.
   - **Commercial use allowed:** only the gated VGGT-1B-Commercial, MapAnything-apache, the Apache-licensed Depth Anything 3 sizes, MegaSaM's own code, DROID-SLAM and COLMAP. Sources are in the Q2 table.
   - **Training data and the target format are clean:** MultiCamVideo is Apache-2.0 and Wan2.2-Fun Control-Camera weights are Apache-2.0, so the training data and the main target generator are both usable.

---

## Q1. Has anyone already done both? Closest works

Searched arXiv, Hugging Face, GitHub, Blender Artists and web search for: camera motion understanding, camera trajectory from video, cinematographic movement classification, camera motion captioning, CameraBench follow-ups, text-to-camera, camera motion transfer, and ML camera tracking add-ons. Reddit could not be searched: the search tool refuses reddit.com for automated clients.

| # | Work | Exact path out? | Named moves out? | Open weights? | Benchmark | How it differs from our plan |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | CameraVLM | Yes, per-frame pose and intrinsics in `.npz` (API) | Yes, JSON motion labels | No | CameraBench, CineTechBench, ShotBench + pose benchmarks | Closest. No DCC export, no released weights, poses far behind VGGT, no zoom-specific eval. |
| 2 | CamChoreo / CamDistill / CamInject | No (pose only as an internal training feature) | Yes, timed segments with type, direction, speed | Yes, 4B and 8B | CamChoreo (micro F1) | Temporal segmentation and speed are already done here. No exportable path. |
| 3 | Geometry-Guided Camera Motion Understanding | Uses 3D foundation model cues internally | Yes, via prompt injection | No code found | CameraMotionVQA | Same "geometry to names" idea, language output only. |
| 4 | CamReasoner | No | Yes | Yes, 7B | CameraBench, ShotBench, RefineShot | VLM only. README says commercial use is prohibited. |
| 5 | ACaM | No | Yes, 17 atomic classes | Yes (collection) | ACaM-Bench | VLM only. Benchmark is CC BY-NC-SA. |
| 6 | CameraBench SFT models | No | Yes | Yes, 7B/32B/72B | CameraBench | The VLM baseline from the source paper. |
| 7 | Cambrian-P | Yes, pose regression head | No | Unverified | VSI-Bench, ScanNet pose | Pose-grounded VLM, no move vocabulary. |
| 8 | Motion Previs Studio | 2D only (pan/tilt/zoom/roll from optical flow) | Partial | Yes, Apache-2.0 app | None | Real product for AI filmmakers. No 6-DoF translation, no benchmark. |
| 9 | MegaSaM / VGGT / Pi3 Blender add-ons | Yes, camera into Blender | No | Depends on backbone | None | Geometry-to-Blender plumbing exists; naming does not. |
| 10 | Text-to-camera generators (E.T., CCD, Director3D, GenDoP, LensCraft, LAMP, CineGEN) | Generate paths from text | Consume names | Mixed | Their own | Opposite direction (text to path). Useful as label vocabularies and data sources. |

### 1. CameraVLM

["When Left Is Right? Camera-Motion Understanding beyond Screen"](https://arxiv.org/abs/2610.17242), 12 Oct 2026.

- **Model.** A 9B model on a Qwen3.5-VL backbone with learnable camera tokens and a camera head. Per [the HTML version](https://arxiv.org/html/2610.17242), the head predicts per-frame translation, quaternion and field of view.
- **Training data.** Labels come from VGGT-Omega, ViPE or ground truth. It trains on CameraBench QA (68K) and ACaM (27K).
- **CameraBench scores**, as reported in that paper (its own protocol):

  | Model | CameraBench score |
  | --- | --- |
  | CameraVLM | 88.2 |
  | CameraBench's 7B model | 70.9 |
  | CamReasoner-7B | 68.0 |
  | ShotVL-7B | 66.1 |
  | ACaM | 63.0 |
  | Qwen3-VL-8B | 62.3 |

- **Pose accuracy.** On Sintel it is roughly on par with VGGT (ATE 0.148 vs 0.172). On ScanNet and TUM it is several times worse.
- **Zoom.** It covers zoom (Appendix A.4 is on zoom vs dolly).
- **Release status.** [The repo](https://github.com/nicolascc1/CameraVLM) lists only "Online demo" and "API" as done. There is a [hosted Space](https://huggingface.co/spaces/NicolasCC/CameraVLM).
- **Licence.** No licence for the code or weights is stated (**unverified**).

**Difference from us:** no export to Blender, FBX, USD or Nuke. No re-shoot validation. Weights not public. Pose quality below dedicated geometry models.

### 2. CamChoreo / CamDistill / CamInject

[CamChoreo](https://arxiv.org/abs/2608.10932): v1 posted 11 Aug 2026, revised 12 Nov 2026.

- **Benchmark.** 4,229 clips with expert temporal annotations over 20 direction-aware labels. Nearly half the segments contain compound motion.
- **Scores.** Same-data SFT reaches 62.1 / 64.1 micro F1 at 4B / 8B. CamDistill adds 5.4 / 3.7 points.
- **Release.** [The CamDistill repo](https://github.com/ddz16/CamDistill) is Apache-2.0. It ships six checkpoints and an inference script that prints JSON with each segment's time span, movement type, direction, speed and special techniques.
- **Gaps.** The CamChoreo benchmark is gated and the training set is not released. Pose is saved only as a 9D encoding for distillation.
- **Licences.** [CamInject-8B](https://huggingface.co/ddz16/CamInject-8B) is tagged Apache-2.0 on a Qwen3-VL-8B base, but it runs VGGT at inference. VGGT and VGGT-Omega weights are non-commercial unless the gated commercial VGGT checkpoint is used. Which checkpoint CamInject loads is **unverified**.

**Difference from us:** this already covers timed segments and speed in language. It does not give an editable camera, and it does not check geometry.

### 3. Geometry-Guided Camera Motion Understanding in VideoLLMs

[arXiv 2603.13119](https://arxiv.org/abs/2603.13119), Mar 2026, CVPR 2026 workshop.

- **Method.** Pulls cues from 3D foundation models, predicts constrained motion primitives with a temporal classifier, and injects them into VideoLLM prompts without retraining.
- **Data.** Released as [a Hugging Face dataset](https://huggingface.co/datasets/fengyee/camera-motion-dataset-and-benchmark). No licence tag found.

**Difference from us:** same core idea of deriving names from geometry, but it outputs language only. No code repo was found.

### 4. CamReasoner

[arXiv 2602.00181](https://arxiv.org/abs/2602.00181v1), Jan 2026.

- **Method.** A 7B model trained with Observation-Thinking-Answer reasoning plus RL.
- **Release.** [Weights](https://huggingface.co/LaurentWu/CamReasoner-7B) are tagged Apache-2.0, but [the repo README](https://github.com/wuhang03/CamReasoner) says commercial use is "strictly prohibited without prior written consent". Treat it as research only.

**Difference from us:** no geometry output.

### 5. ACaM

["Natural Language Camera Movement Understanding"](https://arxiv.org/abs/2607.03043), ECCV 2026.

- **Method.** A two-level cinematographic taxonomy with 17 atomic classes and a fine-tuned 8B Qwen3-VL. The paper reports it beats Gemini 3.1 Pro by 10% (real) and 11% (synthetic).
- **Release.** [ACaM-Bench](https://huggingface.co/datasets/Yuwen2024/ACaM-Bench) is CC BY-NC-SA 4.0: 1,464 real MCQ, 1,179 synthetic MCQ and 1,510 binary items. Weights are in [a collection](https://huggingface.co/collections/Yuwen2024/acam) under a licence that is **unverified**. [The repo](https://github.com/1yuwen/AcaM--Camera-Understanding) has no LICENSE file.

### 6. CameraBench's own fine-tuned models

[Qwen2.5-VL at 7B](https://huggingface.co/chancharikm/qwen2.5-vl-7b-cam-motion), [32B](https://huggingface.co/chancharikm/qwen2.5-vl-32b-cam-motion) and [72B](https://huggingface.co/chancharikm/qwen2.5-vl-72b-cam-motion), listed in [the CameraBench repo](https://github.com/sy77777en/CameraBench). The 7B card's licence field is "other".

### 7. Cambrian-P

[arXiv 2605.22819](https://arxiv.org/abs/2605.22819), May 2026. Adds per-frame camera tokens and a pose regression head to a video MLLM, and reports state-of-the-art streaming pose on ScanNet. No move vocabulary. Weight release is **unverified**.

### 8. Motion Previs Studio

[GitHub repo](https://github.com/wassermanproductions/motion-previs-studio), created Jul 2026, 385 stars, Apache-2.0.

- **What it does.** A desktop app for AI filmmakers. It solves "pan/tilt/zoom/roll" with Lucas-Kanade optical flow and a RANSAC similarity fit, with the subject masked out.
- **Outputs.** Exports `camera_motion.json` and Blender import scripts.
- **Limits.** No translation or parallax, so it cannot tell a dolly from a zoom. No accuracy benchmark.

**Matters as a demand signal and as the incumbent for the creator use case.**

### 9. Geometry-to-Blender add-ons (no move names)

- [CEB_MegaSam](https://github.com/carlosedubarreto/CEB_MegaSam): Oct 2026. Imports the solved MegaSaM camera and point clouds into Blender, and stitches chunks for low VRAM.
- [vggt-blender](https://github.com/xy-gao/vggt-blender): MIT, Jun 2025.
- [pi3-blender](https://github.com/xy-gao/pi3-blender): Jul 2025.

### 10. Text-to-camera and trajectory-language work (opposite direction)

- **Generators:**
  - [E.T. the Exceptional Trajectories](https://arxiv.org/abs/2407.01516)
  - [Cinematographic Camera Diffusion (CCD)](https://arxiv.org/abs/2402.16143)
  - [Director3D](https://arxiv.org/abs/2406.17601)
  - [GenDoP](https://arxiv.org/abs/2504.07083). Its DataDoP set of 29K shots with trajectories and captions is [described in the paper appendix as CC BY 4.0](https://arxiv.org/html/2504.07083v2).
  - [LensCraft](https://arxiv.org/abs/2506.00988)
  - [LAMP](https://arxiv.org/abs/2512.03619)
- **Trajectory-to-text:**
  - [Unveiling the Value of Motion for Cinematic Camera Trajectories](https://arxiv.org/abs/2609.38683): Sep 2026. Argues for a direction-plus-speed representation and does trajectory-to-text alignment. No code link.
  - [Seeing without Pixels](https://arxiv.org/abs/2511.21681): CVPR 2026. Embeds pose trajectories into a language-aligned space.

These give us label vocabularies and possible real-footage pseudo-labels. None reads a video and exports an editable camera.

### Also relevant: camera motion transfer that skips explicit poses

- [CamCloneMaster](https://arxiv.org/abs/2506.03140) (SIGGRAPH Asia 2025) conditions a video model directly on a reference clip. [Checkpoints are public](https://github.com/KwaiVGI/CamCloneMaster); no LICENSE file was detected.
- Adobe shipped [camera motion reference in Firefly](https://www.newscaststudio.com/2025/12/16/adobe-expands-firefly-with-ai-video-editing-tools-and-partner-model-integrations).
- ComfyUI has [a Wan 2.6 reference-to-video workflow](https://blog.comfy.org/p/wan26-reference-to-video).

For AI video creators who only want "copy this move", these skip our tool entirely. Our value there is editability: a path you can tweak, retime and reuse in a 3D tool.

---

## Q2. Geometry baselines to match or beat

| Model | Outputs | Focal / zoom | Speed / GPU (as published) | Code licence | Weights licence | Commercial OK? |
| --- | --- | --- | --- | --- | --- | --- |
| [VGGT](https://arxiv.org/abs/2503.11651) | Extrinsics, intrinsics, depth, point maps, tracks | Per-frame FoV in the pose encoding | Reconstruction "typically under one second" ([README](https://github.com/facebookresearch/vggt)) | VGGT License, commercial except military | [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) CC BY-NC 4.0; [VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial) gated, "vggt-aup-license", no military | Only the commercial checkpoint |
| [VGGT-Omega](https://arxiv.org/abs/2605.15195) (CVPR 2026) | Cameras (extrinsics + intrinsics), depth | Via intrinsics | Peak memory 9.66 GB at 50 frames, 13.37 GB at 100, 43.15 GB at 500 on one A100 ([repo](https://github.com/facebookresearch/vggt-omega)) | [FAIR Noncommercial Research License](https://github.com/facebookresearch/vggt-omega) | [Gated, same licence](https://huggingface.co/facebook/VGGT-Omega) | No |
| [π³ / Pi3X](https://arxiv.org/abs/2507.13347) | Camera-to-world poses, local and global point maps | Not a direct output; recoverable from local point maps | Not stated | BSD-3 | [README](https://github.com/yyfz/Pi3) says Pi3 and Pi3X weights are CC BY-NC 4.0. [Pi3 HF card](https://huggingface.co/yyfz233/Pi3) is tagged bsd-2-clause (conflict, treat as NC). | No |
| [MegaSaM](https://arxiv.org/abs/2412.04463) | Poses, consistent video depth | Optional `--opt_focal`. Whether it supports a changing focal is **unverified**; its zoom AP on CameraBench is 10-11 | Not stated in README | [Apache-2.0](https://github.com/mega-sam/mega-sam) | Depends on Depth Anything and UniDepth weights; their licences are **unverified** | Code yes; check dependencies |
| [CUT3R](https://arxiv.org/abs/2501.12387) | Streaming per-frame point maps and poses | Recoverable | Not stated | [CC BY-NC-SA 4.0](https://github.com/CUT3R/CUT3R) | Same | No |
| [MASt3R-SfM](https://arxiv.org/abs/2409.19152) | SfM poses, points | Per-image | Not stated | [CC BY-NC-SA 4.0](https://github.com/naver/mast3r) | Same | No |
| [MonST3R](https://arxiv.org/abs/2410.03825) | Dynamic point maps, poses, intrinsics | Per-frame | Not stated | [CC BY-NC-SA 4.0](https://github.com/Junyi42/monst3r) | Same | No |
| [Spann3R](https://arxiv.org/abs/2408.16061) | Incremental point maps | Recoverable | Not stated | [CC BY-NC-SA 4.0](https://github.com/HengyiWang/spann3r) | Same | No |
| [DROID-SLAM](https://arxiv.org/abs/2108.10869) | Poses | Needs known intrinsics | Not stated | [BSD-3](https://github.com/princeton-vl/DROID-SLAM) | Same repo | Yes |
| [GLOMAP](https://arxiv.org/abs/2407.20219) | Global SfM | Per-image intrinsics | Not stated | [BSD-3, archived Mar 2026, merged into COLMAP as the global mapper](https://github.com/colmap/glomap) | n/a | Yes |
| [ViPE](https://arxiv.org/abs/2508.10934) (NVIDIA) | Poses, intrinsics, near-metric depth; pinhole, wide-angle and 360 | Yes | 2.7x speed-up in v1.2.0 per [README](https://github.com/nv-tlabs/vipe) | Apache-2.0 for NVIDIA code | Some components non-commercial: Unik3D is BY-NC-SA; MetricTrail is under an NVIDIA non-commercial licence | Only with care |
| [MapAnything](https://arxiv.org/abs/2509.13414) | Intrinsics, cam2world poses, metric scale, depth | Yes | Up to 2,000 views on 140 GB in memory-efficient mode ([repo](https://github.com/facebookresearch/map-anything)) | Apache-2.0 | `map-anything` CC BY-NC; `map-anything-apache` Apache-2.0 | Yes, apache variant |
| [Depth Anything 3](https://arxiv.org/abs/2511.10647) | Extrinsics, intrinsics, depth; streaming mode under 12 GB | Yes | DA3-Streaming "less than 12GB" ([repo](https://github.com/ByteDance-Seed/Depth-Anything-3)) | Apache-2.0 | Small, Base, Metric-Large and Mono-Large are Apache-2.0; Large and Giant are CC BY-NC 4.0 | Yes, smaller sizes |

**Notes for the benchmark plan:**

- **CameraBench converts SfM output to scores without thresholds.** Translation and rotation scores come from relative motion along each axis, and zoom from the ratio of first-frame to last-frame focal length ([paper](https://arxiv.org/html/2504.15376v2)). We can run our geometry the same way and get directly comparable AP.
- **Speed numbers are thin.** Most READMEs give no fps or VRAM figure. Treat speed as something to measure ourselves on one GPU in week 2. Do not quote vendor claims.
- **On dynamic footage,** the [VGGT-Omega](https://www.alphaxiv.org/abs/2605.15195) paper reports Sintel AUC@3° of 35.5 against 22.5 for MegaSaM. That is self-reported and **unverified** by us.
- **Commercial-safe stack for a released tool:** VGGT-1B-Commercial or MapAnything-apache or DA3 Apache sizes for geometry, COLMAP global mapper or DROID-SLAM for a classical check, and MegaSaM's code if its dependency weights clear.

---

## Q3. Move-naming baselines

**What CameraBench evaluated.**

- **VLMs:** Qwen2.5-VL (7B, 32B, 72B), InternVL2.5 (8B, 26B), InternVL3 (8B, 78B), LLaVA-Video-7B, LLaVA-OneVision-7B, InternVideo2-Chat-8B, Tarsier-Recap-7B, mPLUG-Owl3-7B, InternLM-XComposer2.5-7B and GPT-4o.
- **Geometry methods:** MegaSaM, CUT3R, MASt3R, DUSt3R, VGGSfM and COLMAP.

Source: [leaderboard](https://sy77777en.github.io/CameraBench/leaderboard/table.html).

Average AP on camera-centric primitives from that leaderboard:

| Model | Avg AP | Zoom-in AP | Zoom-out AP |
| --- | --- | --- | --- |
| Qwen2.5-VL-72B, CameraBench SFT | 62.8 | 57.1 | 78.5 |
| Qwen2.5-VL-7B, CameraBench SFT | 60.0 | 54.5 | 75.8 |
| MegaSaM | 58.2 | 11.1 | 10.2 |
| CUT3R | 50.9 | 15.9 | 21.3 |
| Qwen2.5-VL-72B (base) | 37.7 | 31.7 | 45.4 |
| InternVL3-78B | 36.8 | 26.4 | 33.4 |
| GPT-4o | 36.4 | 41.7 | 39.3 |
| COLMAP | 27.3 | 13.9 | 14.2 |

The columns are read from the page's table header: 6 translation, 2 zoom, 6 rotation, static, then Avg. The paper text says simple fine-tuning gives Qwen2.5-VL a 1-2x boost and "matches MegaSaM" ([project page](https://linzhiqiu.github.io/papers/camerabench)).

**Best current open models for camera-motion questions (Nov 2026):**

- **CameraVLM (9B).** Highest reported CameraBench score (88.2 in [its paper](https://arxiv.org/html/2610.17242)), but weights are not released.
- **CamDistill-8B / CamInject-8B.**
  - Strongest released family for temporally grounded output ([repo](https://github.com/ddz16/CamDistill)).
  - Code is Apache-2.0. The base [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) is Apache-2.0.
  - CamInject depends on VGGT at inference, see the licence note in Q1.
  - Its CameraBench result is described only as "zero-shot transfer" in the abstract. The number is **unverified**.
- **CameraBench SFT Qwen2.5-VL (7B/32B/72B).** The reference baseline. Licence field is "other".
- **CamReasoner-7B and the ACaM models.** Research use only, or licence unclear.
- **Plain Qwen3-VL-8B.** Scored 62.3 on CameraBench under CameraVLM's protocol ([paper](https://arxiv.org/html/2610.17242)). Apache-2.0, so it is the cleanest commercial baseline.

---

## Q4. CameraBench test set: access, licence, metrics

- **Download.** [syCen/CameraBench](https://huggingface.co/datasets/syCen/CameraBench) is public and not gated.
  - It has one `test` split of about 1,071 clips (5.61 GB) with `labels`, `caption` and video paths.
  - Card licence: **CC BY 4.0**. [The repo](https://github.com/sy77777en/CameraBench) LICENSE is also CC BY 4.0.
  - The card says "We do not publicly redistribute any videos". The clips are internet videos whose rights stay with their owners. The repo ships `download_test_videos.py`.
  - The training split is by request through [a form](https://forms.gle/26LB1rouSZsR29Cx9).
- **Taxonomy** (from [the paper](https://arxiv.org/html/2504.15376v2)):
  - **Motion type:** none, simple, minor, complex.
  - **Steadiness:** five levels.
  - **Translation:** dolly, pedestal, truck.
  - **Rotation:** pan, tilt, roll.
  - **Zoom.**
  - **Object-centric moves:** arc and lead, tail, side, aerial and tilt/pan/arc tracking.
  - **Speed:** slow, regular, fast.
  - **Effects:** dolly-zoom, motion blur.
  - **Scene dynamics.**
- **Metrics.**
  - **Binary primitive classification:** average precision (AP) per primitive.
  - **VQA:** accuracy and question accuracy over 9 skills and 81 sub-tasks, about 10K items.
  - **Captioning:** SPICE and an LLM judge.
  - **Retrieval:** text, video and group scores with CLIPScore, ITMScore and VQAScore.
  - Official evaluation code is in their t2v_metrics fork, linked from [the repo](https://github.com/sy77777en/CameraBench).
- **Can Spheron evaluate on it and report numbers?** Yes, for evaluation only.
  - CC BY 4.0 allows reuse of the annotations with attribution.
  - The project page invites leaderboard submissions ("Please reach out if you wish to add your method").
  - Do not redistribute the clips. Do not train on the test split.
  - CameraVLM trained on 68K CameraBench QA and captions, so any comparison with it on CameraBench needs a note on train/test overlap (**unverified** whether their QA comes only from the training split).

---

## Q5. Camera-controlled video generators that take an exact trajectory

| Model | What it takes | Licence |
| --- | --- | --- |
| Wan2.2-Fun-A14B-Control-Camera, Wan2.2-Fun-5B-Control-Camera, Wan2.1-Fun-V1.1 Control-Camera ([VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)) | A RealEstate10K-style text file, one line per frame, converted to Plücker embeddings. Example presets ship in `asset/` (`Pan_Left.txt`, `Zoom_In.txt`, `CW.txt`, etc.), and the examples load them via `control_camera_txt`. | Code Apache-2.0. Weights Apache-2.0 per the [A14B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control-Camera) and [5B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control-Camera). |
| [CameraCtrl](https://arxiv.org/abs/2404.02101) | Same RealEstate10K per-frame format | [Apache-2.0](https://github.com/hehao13/CameraCtrl) |
| [CameraCtrl II](https://arxiv.org/abs/2503.10592) | Camera trajectories for dynamic scene exploration | Code and licence status **unverified** |
| [ReCamMaster](https://arxiv.org/abs/2503.11647) | 10 preset trajectories via `--cam_type`; custom paths as per-frame extrinsics in the MultiCamVideo `camera_extrinsics.json` style. The public build runs on Wan2.1. | Code [MIT](https://github.com/KwaiVGI/ReCamMaster); weights [Apache-2.0](https://huggingface.co/KlingTeam/ReCamMaster-Wan2.1) |
| [Uni3C](https://arxiv.org/abs/2504.14899) | A rendered guidance video: the target path rendered through a depth-unprojected point cloud. ComfyUI has [a native WanUni3CControlnetApply node](https://docs.comfy.org/built-in-nodes/WanUni3CControlnetApply). | [Apache-2.0](https://github.com/alibaba-damo-academy/Uni3C) |
| [AC3D](https://arxiv.org/abs/2411.18673) | Per-frame camera poses; exact file format **unverified** | [Snap non-commercial research only](https://github.com/snap-research/ac3d) |
| [CamCloneMaster](https://arxiv.org/abs/2506.03140) | A reference video, with no explicit poses | No LICENSE file detected |

A Wan2.2-Fun camera file line has 19 values: `frame fx fy cx cy 0 0` (intrinsics normalised to image size) followed by the 12 values of a 3x4 world-to-camera matrix, row-major. The first line of `Pan_Left.txt` is `0 0.532139961 0.946026558 0.5 0.5 0 0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0`.

**ComfyUI.** [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache-2.0) has:

- Fun-camera nodes that take `CAMERACTRL_POSES`.
- ReCamMaster nodes that read `recam_extrinsics.json`, plus a pose visualiser that also accepts a RealEstate10K intrinsics text file.
- Uni3C nodes that take a rendered latent.

Core ComfyUI also has [a WanCameraEmbedding node](https://docs.comfy.org/built-in-nodes/WanCameraEmbedding) driven by motion presets.

**What to export:**

1. RealEstate10K-style `.txt`. This covers Wan Fun Camera and CameraCtrl, and ReCamMaster's visualiser can read it.
2. ReCamMaster / MultiCamVideo extrinsics JSON.
3. COLMAP text (`cameras.txt`, `images.txt`) for 3D Gaussian splatting and photogrammetry tools.
4. A Blender `.py` importer or `.blend`.
5. USD camera with time-sampled `focalLength`.
6. FBX with a baked camera.
7. Nuke `.chan`.

A rendered point-cloud guide video for Uni3C is a nice extra, not a requirement.

---

## Q6. Demand signals

Reddit could not be checked: the search and fetch tools refuse reddit.com for automated clients. The r/vfx, r/blender, r/comfyui and r/StableDiffusion threads need a manual look (**unverified**).

- **1 Sep 2023, Blender Artists.** [Thread](https://blenderartists.org/t/isnt-there-any-a-i-tool-that-can-extract-camera-motion-from-a-given-video/1480894) asking for an AI tool that takes a video and extracts the camera move for Blender, because the built-in tracker "gets distracted" by moving objects. Replies point to Meshroom, KeenTools GeoTracker and Wonder Dynamics. No one-click answer.
- **4 May 2025, VGGT GitHub.** [Issue #120](https://github.com/facebookresearch/vggt/issues/120): a request to add camera export (Collada) to the demo for setting up scenes in Blender, Maya and 3ds Max. The user pasted their own export code.
- **23 Jun 2025, VGGT GitHub.** [Issue #198](https://github.com/facebookresearch/vggt/issues/198): a user built a VGGT Blender add-on and asks for a packaged tool "for CG artists and 3D modelers".
- **25 Jul 2025, Pi3 GitHub.** [Issue #22](https://github.com/yyfz/Pi3/issues/22): same author, Pi3 Blender add-on.
- **3 Dec 2025, Depth Anything 3 GitHub.** [Issue #128](https://github.com/ByteDance-Seed/Depth-Anything-3/issues/128): "How can I obtain the motion trajectory coordinates of the camera".
- **Dec 2025, Adobe Firefly.** Shipped [camera motion reference](https://www.newscaststudio.com/2025/12/16/adobe-expands-firefly-with-ai-video-editing-tools-and-partner-model-integrations): upload a reference clip to copy its pans, zooms and tilts into a generation. A large vendor productising "copy this camera move".
- **19 May 2026, VGGT-Omega GitHub.** [Issue #21](https://github.com/facebookresearch/vggt-omega/issues/21), with 11 comments, asks how to export Omega results to COLMAP since the old VGGT export scripts may not work.
- **Jul 2026, Motion Previs Studio.** [The app](https://github.com/wassermanproductions/motion-previs-studio) reached 385 stars by offering "Camera only" extraction for AI filmmakers, exporting to Blender and ComfyUI bundles.
- **Oct 2026, CEB_MegaSam.** [The add-on](https://github.com/carlosedubarreto/CEB_MegaSam) wraps MegaSaM camera tracking inside Blender.

**Pattern:** people keep building one-off bridges from research pose models into Blender. Creators want "copy this move" without a full matchmove. Nobody gives them a camera they can edit together with a plain-language description of the move.

---

## Data notes for the plan

- **MultiCamVideo is [Apache-2.0](https://huggingface.co/datasets/KwaiVGI/MultiCamVideo-Dataset)** (now under the KlingTeam org): 13,600 UE5 scenes x 10 cameras, 81 frames at 15 fps, 1280x1280.
  - **Trajectory types:** pan/tilt, translation, arc, random and static, half at constant speed and half at variable speed.
  - **Focal length:** split into four folders (18, 24, 35, 50 mm) per [the ReCamMaster README](https://github.com/KwaiVGI/ReCamMaster). That suggests focal is fixed within a clip (**unverified**). If so it has no in-clip zoom or dolly-zoom, so the procedural Blender renders must supply all zoom and focal-change training data.
- **Mixing in real-footage pseudo-labels.** DataDoP is CC BY 4.0 per [the GenDoP paper](https://arxiv.org/html/2504.07083v2). If we use it, keep source-video rights separate from the annotation licence.
