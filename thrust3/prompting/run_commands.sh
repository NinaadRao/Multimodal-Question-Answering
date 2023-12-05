#!/bin/bash

# Run the commands and append their output to the log file "output.log"
python -u generate_prompt.py pica 1 >> "output.log" 2>&1
python -u generate_prompt.py pica 3 >> "output.log" 2>&1
python -u generate_prompt.py pica 5 >> "output.log" 2>&1
python -u generate_prompt.py pica 10 >> "output.log" 2>&1
python -u generate_prompt.py answer_candidates 1 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_candidates 3 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_candidates 3 5 >> "output.log" 2>&1
python -u generate_prompt.py answer_candidates 5 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_candidates 5 5 >> "output.log" 2>&1
python -u generate_prompt.py answer_candidates 10 5 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 1 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 5 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 10 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware_candidates 1 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware_candidates 3 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware_candidates 3 5 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware_candidates 5 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware_candidates 5 5 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware_candidates 10 5 >> "output.log" 2>&1

python -u generate_prompt.py question_aware 1 >> "output.log" 2>&1
python -u generate_prompt.py question_aware 3 >> "output.log" 2>&1
python -u generate_prompt.py question_aware 5 >> "output.log" 2>&1
python -u generate_prompt.py question_aware 10 >> "output.log" 2>&1
python -u generate_prompt.py question_aware_candidates 1 3 >> "output.log" 2>&1
python -u generate_prompt.py question_aware_candidates 3 3 >> "output.log" 2>&1
python -u generate_prompt.py question_aware_candidates 3 5 >> "output.log" 2>&1
python -u generate_prompt.py question_aware_candidates 5 3 >> "output.log" 2>&1
python -u generate_prompt.py question_aware_candidates 5 5 >> "output.log" 2>&1
python -u generate_prompt.py question_aware_candidates 10 5 >> "output.log" 2>&1


python -u generate_prompt.py pica 1 >> "output.log" 2>&1
python -u generate_prompt.py pica 3 >> "output.log" 2>&1
python -u generate_prompt.py pica 5 >> "output.log" 2>&1
python -u generate_prompt.py pica 10 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 1 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 3 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 5 >> "output.log" 2>&1
python -u generate_prompt.py answer_aware 10 >> "output.log" 2>&1
python -u generate_prompt.py question_aware 1 >> "output.log" 2>&1
python -u generate_prompt.py question_aware 3 >> "output.log" 2>&1
python -u generate_prompt.py question_aware 5 >> "output.log" 2>&1
python -u generate_prompt.py question_aware 10 >> "output.log" 2>&1