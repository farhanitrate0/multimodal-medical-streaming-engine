"""Ingestion and websocket throughput benchmarks."""
import argparse
import asyncio
import base64
import io
import json
import os
import platform
import statistics
import time
import tracemalloc

import numpy as np

OUT = os.path.join("results", "benchmark.json")


def _save(section, data):
    results = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            results = json.load(f)
    results[section] = data
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


def _hardware():
    info = {"cpu": platform.processor(), "python": platform.python_version()}
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def _measure(open_fn, n_pass=1):
    """Peak heap while opening a split and touching every image once."""
    tracemalloc.start()
    t0 = time.perf_counter()
    imgs = open_fn()
    opened = time.perf_counter() - t0
    checksum = 0
    t1 = time.perf_counter()
    for _ in range(n_pass):
        for i in range(len(imgs)):
            checksum += int(np.asarray(imgs[i])[112, 112])  # force a read
    elapsed = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n = len(imgs) * n_pass
    return {"images": n, "open_s": round(opened, 2), "peak_heap_mb": round(peak / 2**20, 1),
            "images_per_s": round(n / elapsed)}


def bench_ingest(split):
    import medmnist
    from memmap_stream import MemmapDataset, build_cache

    build_cache(split)  # one-off, not timed
    in_memory = _measure(lambda: medmnist.PneumoniaMNIST(split=split, size=224).imgs)
    memmap = _measure(lambda: MemmapDataset(split).imgs)
    out = {"split": split, "in_memory_medmnist": in_memory, "memmap": memmap, "hardware": _hardware()}
    print(json.dumps(out, indent=2))
    _save("ingest", out)


def _payload(mode):
    from PIL import Image
    if mode == "medical":
        img = Image.open(os.path.join("samples", "pneumonia_230.png")).convert("RGB")
        fmt, mime = "PNG", "image/png"
    else:
        img = Image.open(os.path.join("docs", "demo1.jpeg")).convert("RGB").resize((640, 480))  # webcam-sized frame
        fmt, mime = "JPEG", "image/jpeg"
    buf = io.BytesIO()
    img.save(buf, fmt)
    return json.dumps({"mode": mode, "image": f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"})


async def _client(url, payload, n, latencies):
    import websockets
    async with websockets.connect(url, max_size=None) as ws:
        for _ in range(n):
            t0 = time.perf_counter()
            await ws.send(payload)
            reply = json.loads(await ws.recv())
            if "error" in reply:
                raise RuntimeError(reply["error"])
            latencies.append(1000 * (time.perf_counter() - t0))


async def _run_level(url, payload, clients, per_client):
    latencies = []
    t0 = time.perf_counter()
    await asyncio.gather(*(_client(url, payload, per_client, latencies) for _ in range(clients)))
    wall = time.perf_counter() - t0
    q = statistics.quantiles(latencies, n=100)
    return {"clients": clients, "requests": len(latencies), "throughput_rps": round(len(latencies) / wall, 1),
            "p50_ms": round(q[49], 1), "p95_ms": round(q[94], 1)}


def bench_server(url, modes, levels, per_client, label):
    out = {"url": url, "per_client_requests": per_client, "hardware": _hardware(), "modes": {}}
    for mode in modes:
        payload = _payload(mode)
        asyncio.run(_run_level(url, payload, 1, 5))  # warm-up
        rows = [asyncio.run(_run_level(url, payload, c, per_client)) for c in levels]
        out["modes"][mode] = rows
        for r in rows:
            print(f"{mode:8s} clients={r['clients']:2d}  {r['throughput_rps']:6.1f} req/s  "
                  f"p50 {r['p50_ms']:6.1f} ms  p95 {r['p95_ms']:6.1f} ms")
    _save(label, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--split", default="train")
    srv = sub.add_parser("server")
    srv.add_argument("--url", default="ws://127.0.0.1:8000/ws/stream")
    srv.add_argument("--modes", nargs="+", default=["medical", "spatial"])
    srv.add_argument("--clients", nargs="+", type=int, default=[1, 4, 8])
    srv.add_argument("--requests", type=int, default=40, help="per client")
    srv.add_argument("--label", default="server", help="key in benchmark.json, e.g. server_unbatched")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    if args.cmd == "ingest":
        bench_ingest(args.split)
    else:
        bench_server(args.url, args.modes, args.clients, args.requests, args.label)


if __name__ == "__main__":
    main()
