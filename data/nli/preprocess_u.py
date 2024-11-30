import os, sys
import json
from pathlib import Path

if __name__== '__main__':

    root_dir = os.path.join(Path.home(), 'sg-llm/data/nli')
    mdl = sys.argv[1]
    datasets = [sys.argv[2]]

    for dataset in datasets:

        split_files1 = [json.load(open(os.path.join(root_dir, f'{dataset}_{mdl}/Z_U_qa_split{i}.json'), 'r')) if os.path.exists(os.path.join(root_dir, f'{dataset}_{mdl}/Z_U_qa_split{i}.json')) else [] for i in range(1, 3)]
        split_files2 = [json.load(open(os.path.join(root_dir, f'{dataset}_{mdl}/Z_U_sam_split{i}.json'), 'r')) if os.path.exists(os.path.join(root_dir, f'{dataset}_{mdl}/Z_U_sam_split{i}.json')) else [] for i in range(1, 3)]
        # read
        data = []
        for split_file1, split_file2 in zip(split_files1, split_files2):
            for idx, (nq, nq2) in enumerate(zip(split_file1, split_file2)):
                data.append(
                    {
                        'question': nq['question'],
                        'answer': nq['answer'],
                        'generated_answer': nq["generated_sequence"],
                        'transformed_answer': nq['transformed_sequence'],
                        'entail_scores': nq['entail_scores'] if 'entail_scores' in nq else None,
                        'labels': None, # TEMP
                        'logprobs': nq['logprobs'],
                        'samples': nq2['samples'],
                        'samples_scores': nq2['samples_scores'],
                        'left_entail_scores': nq['left_entail_scores'] if 'left_entail_scores' in nq else None,
                        'left_samples_scores': nq2['left_samples_scores'] if 'left_samples_scores' in nq2 else None,
                    }
                )
                if len(data) == 30000: break
        print(f'# data = {len(data)}')

        # write
        data_jsonl = '\n'.join([json.dumps(d) for d in data])
        #TODO MJ: directory
        open(os.path.join(root_dir, f'{dataset}_{mdl}/nli.jsonl'), 'w').write(data_jsonl)