import json, glob, os, csv, sys, datetime as dt
arm = sys.argv[1]
lang = {}
for l in open('lang.txt'):
    p = l.split()
    if len(p) >= 2: lang[p[1]] = p[0]
def t(s): return dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
def mins(a, b): return round((t(b) - t(a)).total_seconds() / 60, 2) if a and b else ''
rows = []
for rf in sorted(glob.glob(f'jobs/stage1-{arm}/*__*/result.json')):
    d = os.path.dirname(rf); r = json.load(open(rf))
    task = r['task_name'].split('/')[-1]
    tr = os.path.join(d, 'agent/mini-swe-agent.trajectory.json')
    try: info = json.load(open(tr))['info'] if os.path.exists(tr) else {}
    except Exception: info = {}
    ar = r.get('agent_result') or {}; vr = r.get('verifier_result') or {}
    ae = r.get('agent_execution') or {}; vf = r.get('verifier') or {}; es = r.get('environment_setup') or {}; ag = r.get('agent_setup') or {}
    rows.append(dict(arm=arm, trial=os.path.basename(d), task=task, language=lang.get(task, ''),
        resolved=int((vr.get('rewards') or {}).get('reward', 0) == 1.0),
        exit_status=info.get('exit_status', ''),
        exception=(r.get('exception_info') or {}).get('exception_type', ''),
        model_calls=(info.get('model_stats') or {}).get('api_calls', ''),
        input_tokens=ar.get('n_input_tokens', ''), output_tokens=ar.get('n_output_tokens', ''),
        env_setup_min=mins(es.get('started_at'), es.get('finished_at')),
        agent_setup_min=mins(ag.get('started_at'), ag.get('finished_at')),
        agent_min=mins(ae.get('started_at'), ae.get('finished_at')),
        verifier_min=mins(vf.get('started_at'), vf.get('finished_at')),
        trial_min=mins(r.get('started_at'), r.get('finished_at')),
        started_at=r.get('started_at'), finished_at=r.get('finished_at')))
with open(f'raw/{arm}/trials.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
n = len(rows); res = sum(r['resolved'] for r in rows)
ctx = sum(1 for r in rows if 'ContextWindow' in str(r['exit_status']))
print(f"arm {arm}: {n} trials, {res} resolved ({100*res/max(n,1):.1f}%), {ctx} hit context")
