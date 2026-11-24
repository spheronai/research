"""Needle-in-a-haystack retrieval against an OpenAI-compatible chat endpoint (stdlib + transformers tokenizer).

Three task types, modelled on RULER: single (one needle), multikey (one asked key among 8 needles with
distractor keys), multivalue (one key with 4 values, all must be returned). Haystack is RULER's "noise"
filler. Lengths are exact in tokens of the model's own tokenizer (chat template overhead excluded).

Usage: python niah.py <port> <served_model> <tokenizer_dir> <out_jsonl> [--lengths 8192,32768] [--n 100]
Prints a JSON summary: accuracy per task and length.
"""
import argparse, json, random, string, threading, time, urllib.request
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("port"); p.add_argument("model"); p.add_argument("tokenizer"); p.add_argument("out")
p.add_argument("--lengths", default="8192,32768"); p.add_argument("--n", type=int, default=100)
p.add_argument("--concurrency", type=int, default=16); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
WORDS = [w for w in open(__file__).read().split("WORDLIST:")[-1].split() if w.isalpha()]


def num(rng):
    return str(rng.randint(1000000, 9999999))


def build(task, length, rng):
    keys = rng.sample(WORDS, 8)
    if task == "single":
        needles = [(keys[0], num(rng))]; ask = keys[0]; answer = [needles[0][1]]
    elif task == "multikey":
        needles = [(k, num(rng)) for k in keys]; ask = keys[0]; answer = [needles[0][1]]
    else:  # multivalue
        vals = [num(rng) for _ in range(4)]
        needles = [(keys[0], v) for v in vals]; ask = keys[0]; answer = vals
    sents = [f"One of the special magic numbers for {k} is: {v}. " for k, v in needles]
    q = (f"\n\nWhat {'are all the special magic numbers' if task == 'multivalue' else 'is the special magic number'} "
         f"for {ask} mentioned in the provided text? Answer with the number{'s' if task == 'multivalue' else ''} only.")
    head = "Some special magic numbers are hidden within the following text. Make sure to memorize them. I will quiz you about them afterwards.\n\n"
    budget = length - len(tok.encode(head + q)) - sum(len(tok.encode(s)) for s in sents)
    unit = len(tok.encode(FILLER))
    chunks = [FILLER] * max(1, budget // unit)
    for s in sents:  # needles at random depths
        chunks.insert(rng.randint(0, len(chunks)), s)
    text = head + "".join(chunks) + q
    return text, answer


def ask(prompt):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 64, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", d["usage"]["prompt_tokens"]


jobs = []
for length in map(int, a.lengths.split(",")):
    for task in ("single", "multikey", "multivalue"):
        rng = random.Random(f"{a.seed}-{task}-{length}")
        for i in range(a.n):
            jobs.append((task, length, i) + build(task, length, rng))
results, lock = [], threading.Lock()
queue = list(jobs)


def worker():
    while True:
        with lock:
            if not queue:
                return
            task, length, i, prompt, answer = queue.pop()
        try:
            out, ptoks = ask(prompt); err = None
        except Exception as e:  # recorded, scored as wrong
            out, ptoks, err = "", None, repr(e)
        score = sum(v in out for v in answer) / len(answer)
        with lock:
            results.append({"task": task, "length": length, "i": i, "answer": answer, "output": out,
                            "prompt_tokens": ptoks, "score": score, "error": err})


t0 = time.time()
ts = [threading.Thread(target=worker) for _ in range(a.concurrency)]
[t.start() for t in ts]; [t.join() for t in ts]
with open(a.out, "w") as f:
    for r in sorted(results, key=lambda r: (r["length"], r["task"], r["i"])):
        f.write(json.dumps(r) + "\n")
summary = {}
for r in results:
    k = f"{r['task']}@{r['length']}"
    summary.setdefault(k, []).append(r["score"])
print(json.dumps({"wall_s": round(time.time() - t0, 1),
                  "accuracy": {k: round(sum(v) / len(v), 4) for k, v in sorted(summary.items())},
                  "n": {k: len(v) for k, v in sorted(summary.items())}}))

# WORDLIST: apple river candle marble falcon violet copper harbor lantern meadow orchid pepper quartz
# saddle timber walnut zephyr anchor blossom cactus dolphin ember glacier hazel island jasmine kettle
# lemon mango nectar oyster pebble quiver raven saffron tulip umbrella velvet willow yarrow zinnia
# badger cobalt dragon fennel granite heron indigo jackal kiwi lilac maple nutmeg olive panda
