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
"""Main-table scorer for temporal grounding (Charades-STA): mIoU, recall at IoU 0.3 / 0.5 / 0.7, precision,
recall, and the mean of every numeric meta_info field."""
from cigo.eval.scoring import dataset_info, load_predictions, main, write_results
from cigo.reward import parse as ap

SPLITS = {"charades": "test_set_grpo"}


def get_dataset_info(name):
    return dataset_info(name, SPLITS[name]) if name in SPLITS else None


def calc_eval_result(output_path, gt_file, num_chunks):
    prediction_set = {}
    parse_fail_cnt = 0
    for sample in load_predictions(output_path, gt_file, num_chunks):
        parsed = ap.tg_for_eval(sample['pred'])
        pred_list = list(parsed.value) if parsed.value is not None else [-1, -1]
        gt = ap.gold_span(sample['answer'])
        # An unreadable gold span counts as a parse fail.
        iou, precision, recall = ap.cal_iou_precision_recall(gt, pred_list) if gt else (-1, -1, -1)
        if iou == -2:
            print(f"Parsing failed: ans={sample['answer']}, pred={sample['pred']}, gt error, jump it")
            continue
        elif iou == -1:
            print(f"Parsing failed: ans={sample['answer']}, pred={sample['pred']}")
            parse_fail_cnt += 1
            iou = precision = recall = 0
        prediction_set[sample['id']] = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'recall@0.3': 1 if iou >= 0.3 else 0,
            'recall@0.5': 1 if iou >= 0.5 else 0,
            'recall@0.7': 1 if iou >= 0.7 else 0,
            'answered': 0.0 if parsed.abstained else 1.0,
            'confident': 1.0 if parsed.confident else 0.0,
            'parse_rule': parsed.rule,
            'parse_scope': parsed.scope,
            'in_answer_tag': float(parsed.in_answer_tag),
            **sample
        }

    scores = []
    for result in prediction_set.values():
        scores += [("recall@0.3", result['recall@0.3']), ("recall@0.5", result['recall@0.5']),
                   ("recall@0.7", result['recall@0.7']), ("miou", result['iou']),
                   ("precision", result['precision']), ("recall", result['recall'])]
        # bools count as numbers
        scores += [(k, v) for k, v in result.get('meta_info', {}).items() if isinstance(v, (int, float))]
    write_results(output_path, prediction_set,
                  [f"total samples = {len(prediction_set)}", f"parse fail = {parse_fail_cnt}"], scores)


if __name__ == "__main__":
    main(get_dataset_info, calc_eval_result)
