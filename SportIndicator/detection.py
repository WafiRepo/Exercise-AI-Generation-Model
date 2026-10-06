import json, argparse, tqdm, time, os, re
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
def read_template(prompt_fp):
    text_template = open(prompt_fp).read()
    return text_template

def read_data(predict_path):
    """
    Membaca data dari file JSON atau file teks biasa.
    Jika file adalah teks biasa, akan dikonversi ke format JSON yang diharapkan.
    
    Format JSON yang diharapkan:
    [
        {"system_output": "instruction text 1"},
        {"system_output": "instruction text 2"},
        ...
    ]
    """
    if not os.path.exists(predict_path):
        raise FileNotFoundError(f"File not found: {predict_path}")
    
    # Cek apakah file kosong
    if os.path.getsize(predict_path) == 0:
        raise ValueError(f"File is empty: {predict_path}")
    
    try:
        # Coba baca sebagai JSON
        with open(predict_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Jika file kosong setelah strip
        if not content:
            raise ValueError(f"File is empty: {predict_path}")
        
        # Coba parse sebagai JSON
        try:
            predictions = json.loads(content)
            
            # Validasi format
            if isinstance(predictions, list):
                # Format list of dicts
                if len(predictions) > 0 and isinstance(predictions[0], dict):
                    # Pastikan ada 'system_output' key
                    if 'system_output' not in predictions[0]:
                        raise ValueError(
                            f"JSON format error: Expected 'system_output' key in dictionary. "
                            f"Found keys: {list(predictions[0].keys())}"
                        )
                    return predictions
                else:
                    raise ValueError("JSON format error: Expected list of dictionaries")
            elif isinstance(predictions, dict):
                # Format single dict atau dict dengan multiple entries
                # Coba konversi ke list format
                if 'system_output' in predictions:
                    return [predictions]
                else:
                    # Mungkin format dict dengan keys sebagai filenames
                    results = []
                    for key, value in predictions.items():
                        if isinstance(value, str):
                            results.append({"system_output": value})
                        elif isinstance(value, dict) and 'system_output' in value:
                            results.append(value)
                        else:
                            results.append({"system_output": str(value)})
                    return results
            else:
                raise ValueError(f"JSON format error: Expected list or dict, got {type(predictions)}")
                
        except json.JSONDecodeError as e:
            # Jika bukan JSON, coba baca sebagai file teks biasa
            print(f"ℹ Info: File is not valid JSON. Reading as plain text and converting to expected format: {predict_path}")
            
            # Baca sebagai teks dan konversi ke format JSON
            with open(predict_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            if not text_content:
                raise ValueError(f"File is empty: {predict_path}")
            
            # Konversi teks ke format JSON yang diharapkan
            print(f"✓ Converted text file to JSON format (1 entry)")
            return [{"system_output": text_content}]
            
    except Exception as e:
        raise ValueError(f"Error reading file {predict_path}: {str(e)}")

def acc(args, filename, metric):
    output_filename = metric + '_Detection_{{filename}}.json'

    output_filename = output_filename.replace('{{filename}}',filename)
    output_filepath = os.path.join(args.output,output_filename)
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    if not os.path.exists(output_filepath):
        raise FileNotFoundError(
            f"Evaluation results file not found: {output_filepath}\n"
            f"Please run g_eval first to generate this file."
        )
    
    with open(output_filepath, 'r') as f:
        datas = json.load(f)

    acc_shot_count = 0
    acc_score = 0

    score_name = metric + '_Detection_score'
    has_ground_truth = False
    
    for data in datas:
        # Check if ground truth score exists (for accuracy calculation)
        # If not, skip accuracy calculation (only detection score available)
        if "score" in data:
            has_ground_truth = True
            if isinstance(data["score"], list):
                bestscore = max(data["score"])
            else:
                bestscore = data["score"]
            
            # Only count if both detection score > 0 and ground truth score > 1
            if score_name in data and data[score_name] != 0 and bestscore > 1:
                acc_shot_count += 1
                acc_score += bestscore
        else:
            # If no ground truth score, we can't calculate accuracy
            # This is expected when evaluating new predictions without ground truth
            # Just skip this entry
            continue
    
    # If no ground truth scores found, warn user
    if not has_ground_truth:
        print(f"⚠ Warning: No ground truth 'score' found in evaluation results.")
        print(f"  Accuracy calculation requires ground truth scores.")
        print(f"  Only detection scores will be reported.")
    
    return acc_score, acc_shot_count

def g_eval(args, summeval, prompt, api_key, filename, metric, provider='anthropic'):
    """
    G-Eval function dengan support untuk Anthropic dan OpenAI
    
    Args:
        args: Argument parser object
        summeval: List of evaluation instances
        prompt: Prompt template
        api_key: API key untuk provider
        filename: Output filename
        metric: Metric name (BodyPart, Error, etc.)
        provider: 'anthropic' atau 'openai' (default: 'anthropic')
    """
    new_json = []
    count, ignore, all_score = 0, 0, 0

    score_name = metric + '_Detection_score'
    prompt_name = metric + '_Detection_prompt'

    for instance in tqdm.tqdm(summeval):
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Instruction}}', system_output)
        instance[prompt_name] = cur_prompt
        
        try:
            if provider == 'openai':
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI library not installed. Install with: pip install openai")
                
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",  # atau "gpt-4-turbo", "gpt-3.5-turbo"
                    messages=[{"role": "user", "content": cur_prompt}],
                    max_tokens=5,
                    temperature=0
                )
                text_value = response.choices[0].message.content
                print(f"OpenAI response: {text_value}")
                
            else:  # anthropic (default)
                if not ANTHROPIC_AVAILABLE:
                    raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
                
                client = anthropic.Anthropic(api_key=api_key)
                _response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    messages=[{"role": "user", "content": cur_prompt}],
                    max_tokens=5
                )
                content = _response.content
                print(f"Claude response: {content}")
                text_value = content[0].text
            
            # Extract score (1-5) dari response
            match = re.search(r'\d+', text_value)
            if match:
                score = int(match.group())
                # Clamp score to 1-5 range
                score = max(1, min(5, score))
                instance[score_name] = score
            else:
                print(f"⚠ Warning: Could not extract score from response: {text_value}")
                instance[score_name] = 0  # Default to 0 if can't parse
            
            new_json.append(instance)
            count += 1
            all_score += instance[score_name]

        except Exception as e:
            print(f"Error occurred: {e}")
            ignore += 1
    
    output_filename = metric + '_Detection_{{filename}}.json'
    output_filename = output_filename.replace('{{filename}}', filename)

    output_filepath = os.path.join(args.output, output_filename)
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    with open(output_filepath, 'w') as f:
        json.dump(new_json, f, indent=4)
    
    print(f"✓ Saved evaluation results to: {output_filepath}")
    
    if count > 0:
        return all_score/count, all_score
    else:
        return 0, 0