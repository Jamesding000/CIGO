# Data

`data/train/` holds the 4,800 training questions used in the paper. The steps below build a training set the same
way from the source datasets.

## 1. Source the videos and questions

We follow [ThinkingWithVideos](https://github.com/zhang9302002/ThinkingWithVideos) for the videos and questions of
LongVideo-Reason, Video-R1 and Charades-STA, and use [CG-Bench](https://github.com/CG-Bench/CG-Bench) for long
videos. Convert each source into a JSON list of questions:

```json
{"id": "...", "video": "<video id>.mp4", "text": "<question with options A./B./...>", "solution": "B", "duration": 120.0, "data_source": "video_r1/multiple_choice"}
{"id": "...", "video": "<video id>.mp4", "text": "<grounding query>", "solution": [10.9, 20.18], "duration": 20.18, "data_source": "charades_tool"}
```

`solution` is the option letter for multiple choice and `[start, end]` in seconds for temporal grounding.

## 2. Extract frames

```bash
python -m cigo.data.extract_frames --input-dir <videos> \
    --output-dir $CIGO_DATA_ROOT/<dataset>/video_14400frames_fps2      # 2 fps, one directory per video
```

| Dataset | Frame directory |
|---|---|
| LongVideo-Reason | `$CIGO_DATA_ROOT/longvideo-reason/video_14400frames_fps2` |
| CG-Bench | `$CIGO_DATA_ROOT/cgbench_mini/video_14400frames_fps2` |
| Video-R1, Charades-STA | `$CIGO_DATA_ROOT/charades/video_14400frames_fps2` |

## 3. Filter

Optionally draw a random subset of a source as its candidates, e.g.
`python -c "import json,random,sys; r=json.load(open(sys.argv[1])); random.seed(0); json.dump(random.sample(r,int(sys.argv[3])),open(sys.argv[2],'w'))" source.json candidates.json 5000`.

**Stage 1, blind filter.** The model answers each question 4 times without the video; a question answered above
chance (binomial test for multiple choice, random-interval test for grounding, α = 0.01) is removed.

```bash
python -m cigo.data.blind_filter --model $CIGO_MODEL_DIR/Qwen3-VL-8B-Thinking \
    --input candidates.json --output nonblind.json --work-dir work/blind --num-gpus 4
```

**Stage 2, answerability filter.** The model answers each question 8 times with the video; a question answered
all correctly or all incorrectly is removed (grounding: IoU spread at most 0.02).

```bash
python -m cigo.data.answerability_filter --model $CIGO_MODEL_DIR/Qwen3-VL-8B-Thinking \
    --input nonblind.json --frame-dir $CIGO_DATA_ROOT/<dataset>/video_14400frames_fps2 \
    --output filtered.json --work-dir work/answerability --num-gpus 4
```

Each filter writes the kept questions to `--output` and every question's statistics to `<output>.stats.json`.
Rollouts are saved in `--work-dir`, so an interrupted run resumes.

## 4. Select

From the filtered candidates of the four sources we select 4,800 questions:

| File | Source | `data_source` | Task | Questions |
|---|---|---|---|---|
| `data/train/longvideo_reason.json` | LongVideo-Reason | `video_r1/multiple_choice` | multiple choice | 1,449 |
| `data/train/cg_bench.json` | CG-Bench | `video_r1/multiple_choice` | multiple choice | 1,608 |
| `data/train/video_r1.json` | Video-R1 (Charades videos) | `video_r1/multiple_choice` | multiple choice | 303 |
| `data/train/charades_sta.json` | Charades-STA | `charades_tool` | temporal grounding | 1,440 |

## License

The LongVideo-Reason, Video-R1 and Charades-STA questions come from a dataset released under CC BY-NC-SA 4.0; the
files in `data/train/` are derived from it and carry the same license.
