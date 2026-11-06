"""Last balanced boxed answer, graded by Math-Verify inside an outer timed process."""
import json
import re
import subprocess
import sys
from pathlib import Path


def last_box(text):
    matches=list(re.finditer(r'\\boxed\s*\{',text))
    if not matches:return None,'missing-box'
    start=matches[-1].end();depth=1;i=start
    while i<len(text):
        if text[i]=='\\' and i+1<len(text) and text[i+1] in '{}':i+=2;continue
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start:i],'parsed-box'
        i+=1
    return None,'unclosed-last-box'


def grade_in_process(text,gold):
    from math_verify import LatexExtractionConfig,parse,verify
    box,status=last_box(text)
    result={'last_box':box,'extraction_status':status,'correct':False}
    if box is None:return result
    config=[LatexExtractionConfig()]
    parse_value=lambda value:parse('$\\boxed{'+value+'}$',extraction_config=config,parsing_timeout=None,raise_on_error=True)
    reference=parse_value(gold);prediction=parse_value(box)
    if not reference:raise ValueError('Gold answer could not be parsed')
    result.update(correct=bool(verify(reference,prediction,timeout_seconds=None)),gold_parsed=[str(x) for x in reference],prediction_parsed=[str(x) for x in prediction])
    return result


def grade(text,gold,timeout=15):
    """Windows-safe timeout: isolate the parser, rather than nesting multiprocessing."""
    completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--stdin'],input=json.dumps({'text':text,'gold':gold}),
        text=True,encoding='utf-8',capture_output=True,timeout=timeout,check=True)
    return json.loads(completed.stdout)


if __name__=='__main__':
    request=json.load(sys.stdin)
    print(json.dumps(grade_in_process(request['text'],request['gold'])))
