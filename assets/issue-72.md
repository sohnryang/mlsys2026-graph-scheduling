---
source: github
repo: owner/repo
issue_number: 72
issue_title: "Follow up on testing environment"
issue_url: https://github.com/yarongmu-google/MLSys/issues/72
exported_at: 2026-04-23T09:44:20Z
---

# Issue #72: Follow up on testing environment

## Original post
- author: jerryyiransun
- created_at: 2026-04-08T01:03:07Z
- url: https://github.com/yarongmu-google/MLSys/issues/72

<comment>
Thanks for the clarification in #54. I understand the goal is to avoid requiring access to a specific cloud provider.

My remaining concern is around how teams can reliably validate timeout compliance before final submission. Since Track A scoring depends on strict runtime limits, differences in CPU microarchitecture, clock speed, cache size, memory bandwidth, and supported instruction sets can significantly affect execution time, even across systems that are all nominally “8-core Linux workstations.”

Without a closer reference point for the evaluation hardware, it’s difficult to know whether a solution that passes locally will still fit within the timeout in the official harness.
</comment>

---

## Comment 1
- author: Gaurav-Shah05
- created_at: 2026-04-10T05:47:22Z
- url: https://github.com/yarongmu-google/MLSys/issues/72#issuecomment-4221156951

<comment>
I had the same question. The readme just mentions "The contest organizers will execute each team's binary across the twenty withheld benchmarks on a dedicated 8-core Linux workstation with 32GB of RAM". As the solver's latency tested on a local machine might vary from when tested on the evaluation hardware, it would help to get more details on the hardware and the compilation targets for the compiled binary to be submitted.
</comment>

---

## Comment 2
- author: jerryyiransun
- created_at: 2026-04-22T03:11:50Z
- url: https://github.com/yarongmu-google/MLSys/issues/72#issuecomment-4293272002

<comment>
@yarongmu-google since the competition deadline is coming up, can we please get an update on this so we can test that our solution fits under the timeout
</comment>

---

## Comment 3
- author: yarongmu-google
- created_at: 2026-04-22T03:47:40Z
- url: https://github.com/yarongmu-google/MLSys/issues/72#issuecomment-4293374892

<comment>
Fair concern, and thanks for the patience on this.

**What we can lock in:**

- **OS / architecture**: Ubuntu 22.04 LTS, `x86_64`. Binaries should be statically linked or compiled against glibc ≤ the Ubuntu 22.04 version to avoid ABI mismatches.
- **Instruction-set baseline**: target **`x86-64-v3`** (AVX2, BMI2, FMA3). This is conservative for all x86_64 server CPUs since ~2015. If you compile with `-march=x86-64-v3` (GCC 11+ / Clang 12+), you'll run on the evaluation host.
- **Cores / memory**: 8 cores, 32 GB RAM, as stated in the README. Your binary can thread up to 8-way; beyond that it's just timesharing.
- **Timing**: wall-clock. Your binary is launched as a subprocess and hard-killed at its tier's timeout (2 s / 5 s / 15 s / 30 s / 60 s / 120 s for tiers 1–6). No CPU-time limit, no memory cgroup, no namespacing applied to your process.
- **Invocation**: `./mlsys <benchmark.json> <output.json>` with the binary's directory as cwd. No stdin/stdout required.

**Easy local-verification recipe.** Rather than target a specific GCE instance (which would force GCP access on everyone — see #54), run under Docker with the declared 8-core / 32 GB limits. That's portable across Mac, Windows, and Linux hosts, something like this:

```bash
# Build your binary in a matching environment
docker run --rm --platform=linux/amd64 \
    -v $(pwd):/work -w /work \
    ubuntu:22.04 bash -c \
    "apt-get update && apt-get install -y build-essential g++-11 && \
     g++-11 -O2 -march=x86-64-v3 -o mlsys your_source.cc"

# Run the benchmarks under the advertised resource cap
docker run --rm --platform=linux/amd64 \
    --cpus=8 --memory=32g \
    -v $(pwd):/work -w /work \
    ubuntu:22.04 ./mlsys benchmark.json output.json
```

`--cpus=8 --memory=32g` enforces exactly what the README promises. If your binary runs within its tier's timeout on a reasonably fast host under that cap, it will run on our grading host (which is faster than the advertised baseline; we expect a margin).

**Anytime strategy.** If your solver does any search, write a valid baseline solution JSON to disk before the expensive part, and overwrite atomically on each improvement. If we SIGTERM you at the deadline, whatever you last wrote is what we score. Several submissions that make it through tight timeouts use this pattern — it's a robust design regardless of host differences.

**Why we don't pin a specific machine**: the 8-core / 32 GB / Ubuntu 22.04 / `x86-64-v3` target is the spec we grade against. Our actual grading host has more resources than that, so timeout compliance on a Docker-capped 8-core box is a safe lower bound. If your binary fits in the advertised envelope, it fits in ours.

Let me know if that unblocks you, or if there's a specific instruction-set or timing detail still unclear.
</comment>

---

## Comment 4
- author: yarongmu-google
- created_at: 2026-04-22T03:48:16Z
- url: https://github.com/yarongmu-google/MLSys/issues/72#issuecomment-4293376501

<comment>
I will resolve this for now. Please open a new issue, referring this one, if the above doesn't make sense. 
</comment>

---

