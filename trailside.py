"""Trailside Birder: record -> identify (BirdNET) -> explain (local LLM) -> speak.

Everything runs locally. Needs Ollama running (https://ollama.com) with a model pulled,
e.g.  ollama pull gemma3:1b
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys, tempfile, datetime
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
LIFE_LIST = os.path.expanduser("~/.trailside_life_list.json")

VOICES = {
    "journal": "warm field-journal style, like a naturalist's notebook",
    "kid": "friendly and simple, for a curious 8 year old",
    "expert": "concise and technical, like an experienced ornithologist",
}


def record(path, seconds):
    import sounddevice as sd, soundfile as sf
    rate = 48000  # BirdNET expects 48 kHz
    print(f"Listening for {seconds}s...")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, rate)


def identify(path, min_conf, lat=None, lon=None):
    out = tempfile.mkdtemp()
    cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", path,
           "-o", out, "--rtype", "csv", "--min_conf", str(min_conf)]
    if lat is not None and lon is not None:
        cmd += ["--lat", str(lat), "--lon", str(lon)]
    subprocess.run(cmd, check=True)
    best = {}
    for f in glob.glob(os.path.join(out, "**", "*.csv"), recursive=True):
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = row.get("Common name") or row.get("common_name")
                conf = float(row.get("Confidence") or row.get("confidence") or 0)
                if name and conf > best.get(name, 0):
                    best[name] = conf
    if not best:
        return None
    name = max(best, key=best.get)
    return name, best[name]


def explain(name, conf, voice, model):
    hedge = "It is only a possible match, so say so." if conf < 0.7 else ""
    prompt = (f"A walker just heard a {name} (confidence {conf:.0%}). "
              f"In two short sentences, in a {VOICES[voice]}, say what it is and "
              f"one thing to listen for in its song. {hedge} No emoji, no lists.")
    r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def speak(text):
    for tts in ("espeak-ng", "espeak", "say"):
        if shutil.which(tts):
            subprocess.run([tts, text])
            return
    print("(No text-to-speech found. Install espeak-ng for spoken output.)")


def log(name, conf, note):
    data = []
    if os.path.exists(LIFE_LIST):
        with open(LIFE_LIST) as f:
            data = json.load(f)
    data.append({"species": name, "confidence": round(conf, 2),
                 "date": datetime.datetime.now().isoformat(timespec="minutes"), "note": note})
    with open(LIFE_LIST, "w") as f:
        json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", help="analyze an existing audio file instead of recording")
    p.add_argument("--seconds", type=int, default=15)
    p.add_argument("--voice", choices=VOICES, default="journal")
    p.add_argument("--model", default="gemma3:1b")
    p.add_argument("--min-conf", type=float, default=0.4)
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--quiet", action="store_true", help="print only, no speech")
    a = p.parse_args()

    path = a.file or os.path.join(tempfile.gettempdir(), "trailside.wav")
    if not a.file:
        record(path, a.seconds)
    result = identify(path, a.min_conf, a.lat, a.lon)
    if not result:
        msg = "Nothing clear that time. Try again in a quieter moment."
        print(msg); (None if a.quiet else speak(msg)); return
    name, conf = result
    text = explain(name, conf, a.voice, a.model)
    print(f"{name} ({conf:.0%})\n{text}")
    log(name, conf, text)
    if not a.quiet:
        speak(text)


if __name__ == "__main__":
    main()
