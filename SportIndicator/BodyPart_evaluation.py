import json, argparse, tqdm, time, os, re
from dotenv import load_dotenv
from detection import read_template, read_data, acc, g_eval
# BodyPart
indicator = 'BodyPart'

# Usage :
# $ python SportIndicator/BodyPart_evaluation.py

if __name__ == '__main__':
    load_dotenv()
    argparser = argparse.ArgumentParser()

    # BodyPart
    argparser.add_argument('--prompt_fp', type=str, default='./SportIndicator/GEval_template/GEval_Bodypart_template.txt')
    # Skating
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_epoch_135.json')
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_instruction_by_GPT4-o.json')
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_llama32_skating.json')      
    # argparser.add_argument('--output', type=str, default="./results/boxing_aligned/.")
    # argparser.add_argument('--output', type=str, default="./results/boxing_llama/.")
    # argparser.add_argument('--output', type=str, default="./results/boxing_gpt/.")
    # argparser.add_argument('--output', type=str, default="./results/GT_BX/.")
    argparser.add_argument('--output', type=str, default="./results/GT_FS/.")
    # Boxing
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_boxing_epoch70.json')
    # argparser.add_argument('--predict', type=str, default="./results/boxing_aligned/geval/geval_epoch_115.json")
    # argparser.add_argument('--predict', type=str, default="./results/boxing_llama/geval/geval_epoch_31.json")
    # argparser.add_argument('--predict', type=str, default="./results/boxing_gpt/geval/geval_epoch_31.json")
    # argparser.add_argument('--predict', type=str, default="./results/GT_BX/geval/results.json")
    argparser.add_argument('--predict', type=str, default="./results/GT_FS/geval/results.json")
    # Tambahkan argument untuk provider
    argparser.add_argument('--provider', type=str, default='anthropic',
                          choices=['anthropic', 'openai'],
                          help='LLM provider: anthropic (Claude) or openai (GPT)')
    argparser.add_argument('--api_key', type=str, default=None,
                          help='API key (if not provided, will use env var)')
    args = argparser.parse_args()

    # Get API key dari argument atau environment variable
    if args.api_key:
        api_key = args.api_key
    elif args.provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Set it with: export OPENAI_API_KEY=your_key")
    else:  # anthropic
        api_key = os.getenv("ANTHROPIC_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_KEY not found in environment variables. Set it with: export ANTHROPIC_KEY=your_key")
    
    prompt      = read_template(args.prompt_fp)

    Scores      = {}
    filename    = os.path.basename(args.predict)
    filename    = os.path.splitext(filename)[0]

    all_filename = indicator + f'_Detection_avg{filename}.json'
    all_filepath = os.path.join(args.output, all_filename)
        
    file_path   = os.path.join(args.predict, filename)
            
    # G-eval
    results             = read_data(args.predict)
    avg_score, score    = g_eval(args, results, prompt, api_key, filename, indicator, provider=args.provider)
    avg_score_name      = indicator + '_Detection_avg_score'
    score_name          = indicator + '_Detection_score'

    Scores[filename] = {avg_score_name : avg_score,
                        score_name : score}

    # Accuracy
    acc_score_name      = indicator + '_acc_score'
    acc_shot_count_name = indicator + '_acc_shot_count'
    acc_name             = indicator + '_acc'
    acc_score, acc_shot_count = acc(args, filename, indicator)
    
    Scores[acc_score_name] = acc_score
    Scores[acc_shot_count_name] = acc_shot_count
    if acc_shot_count > 0:
        Scores[acc_name] = acc_score/acc_shot_count
    else:
        Scores[acc_name] = 0.0
        print("⚠ Warning: No valid entries for accuracy calculation. Accuracy set to 0.")

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    with open(all_filepath, 'w') as f:
        json.dump(Scores, f, indent=4)
    
    print(f"✓ Saved final scores to: {all_filepath}")