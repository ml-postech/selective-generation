import os, sys
import json
from pathlib import Path

if __name__== '__main__':

    root_dir = os.path.join('data/nli')
    mdl = sys.argv[1]
    datasets = [sys.argv[2]]

    for dataset in datasets:
            
        split_files1 = [json.load(open(os.path.join(root_dir, f'{dataset}_{mdl}/Z_E_qa_split{i}.json'), 'r')) if os.path.exists(os.path.join(root_dir, f'{dataset}_{mdl}/Z_E_qa_split{i}.json')) else [] for i in range(1, 3)]
        split_files2 = [json.load(open(os.path.join(root_dir, f'{dataset}_{mdl}/Z_E_sam_split{i}.json'), 'r')) if os.path.exists(os.path.join(root_dir, f'{dataset}_{mdl}/Z_E_sam_split{i}.json')) else [] for i in range(1, 3)]
        label_dict = {
            'entail': 1,
            'neutral': 1,
            'contradict': 0,
            None: None,
        }
        data1 = []
        data2 = []
        for split_file1, split_file2 in zip(split_files1, split_files2):
            for nq, nq2 in zip(split_file1, split_file2):
                if nq['label']:
                    data1.append(
                        {
                            'question': nq['question'],
                            'answer': nq['answer'],
                            'generated_answer': nq["generated_sequence"],
                            'transformed_answer': nq['transformed_sequence'],
                            'entail_scores': nq['entail_scores'] if 'entail_scores' in nq else None,
                            'labels': label_dict[nq['label']],
                            'logprobs': nq['logprobs'],
                            'samples': nq2['samples'],
                            'samples_scores': nq2['samples_scores'] if 'samples_scores' in nq2 else None,
                            # 'left_entail_scores': nq['left_entail_scores'] if 'left_entail_scores' in nq else None,
                            # 'left_samples_scores': nq2['left_samples_scores'] if 'left_samples_scores' in nq2 else None,
                        }
                    )
                # deprecated
                else:
                    data2.append(
                        {
                            'question': nq['question'],
                            'answer': nq['answer'],
                            'generated_answer': nq["generated_sequence"],
                            'transformed_answer': nq['transformed_sequence'],
                            'entail_scores': nq['entail_scores'],
                            'labels': None,
                        }
                    )

        print(f'# Labeled = {len(data1)}, Not labeled = {len(data2)}')

        # write
        data_jsonl = '\n'.join([json.dumps(d) for d in data1])
        open(os.path.join(root_dir, f'{dataset}_{mdl}/nli_labeled.jsonl'), 'w').write(data_jsonl)
        # data_jsonl = '\n'.join([json.dumps(d) for d in data2])
        # open(os.path.join(root_dir, 'nq_labels/nli.jsonl'), 'w').write(data_jsonl)