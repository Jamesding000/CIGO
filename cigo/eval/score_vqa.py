#    Copyright 2024 Flash-VStream Authors
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Main-table scorer for the multiple-choice benchmarks: accuracy overall and per question type."""
from cigo.eval.scoring import dataset_info, load_predictions, main, write_results
from cigo.reward import parse as ap

# Ground-truth file stem per benchmark.
SPLITS = {"videomme": "test_set_grpo", "mlvu": "dev_set_grpo", "lvbench": "test_set_grpo",
          "longvideobench": "val_set_grpo", "videommmu": "test_set_grpo"}


def get_dataset_info(name):
    return dataset_info(name, SPLITS[name]) if name in SPLITS else None


def calc_eval_result(output_path, gt_file, num_chunks):
    prediction_set = {}
    parse_fail_cnt = 0
    invalid_format_cnt = 0
    for sample in load_predictions(output_path, gt_file, num_chunks):
        try:
            pred_text = sample['pred']
            if '<answer>' not in pred_text or '</answer>' not in pred_text:
                invalid_format_cnt += 1
            parsed = ap.mcq_for_eval(pred_text, sample["question"])
            correct = parsed.value and parsed.value == ap.gold_letter(str(sample["answer"]), sample["question"])
            prediction_set[sample['id']] = {
                'acc': 1.0 if correct else 0.0,
                'answered': 1.0 if parsed.value else 0.0,
                'confident': 1.0 if parsed.confident else 0.0,
                'parse_rule': parsed.rule,
                'parse_scope': parsed.scope,
                **sample
            }
        except Exception as e:
            parse_fail_cnt += 1
            print(f"[main] parse fail: {e}")

    scores = []
    for result in prediction_set.values():
        scores.append(("acc", result['acc']))
        typ = result.get('question_type')
        if isinstance(typ, str):
            typ = [typ]
        if isinstance(typ, list):
            scores += [(t, result['acc']) for t in typ]
    write_results(output_path, prediction_set,
                  [f"total samples = {len(prediction_set)}", f"parse fail = {parse_fail_cnt}",
                   f"invalid format (no <answer> tags) = {invalid_format_cnt}"], scores)


if __name__ == "__main__":
    main(get_dataset_info, calc_eval_result)
