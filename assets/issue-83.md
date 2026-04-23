---
source: github
repo: owner/repo
issue_number: 83
issue_title: "Follow up on testing environment"
issue_url: https://github.com/yarongmu-google/MLSys/issues/83
exported_at: 2026-04-23T09:45:46Z
---

# Issue #83: Follow up on testing environment

## Original post
- author: jerryyiransun
- created_at: 2026-04-22T18:34:36Z
- url: https://github.com/yarongmu-google/MLSys/issues/83

<comment>
One quick follow-up to #72: is the sample `g++-11` command only an example, or are submissions expected to be compiled that way? We are using CMake rather than a single-file compile command, but our build still targets Ubuntu 22.04 and `x86-64-v3` and produces the required mlsys binary.
</comment>

---

## Comment 1
- author: yarongmu-google
- created_at: 2026-04-22T19:36:49Z
- url: https://github.com/yarongmu-google/MLSys/issues/83#issuecomment-4299376325

<comment>
Thanks for the question.

The g++-11 command in #72 was illustrative, not prescriptive. You don't need to use that exact invocation.           
                                                                                                                       
What we actually care about is the runtime target, not how you produce the binary:
- The submitted mlsys binary must run on Ubuntu 22.04 LTS, x86-64-v3.
- It must execute successfully from the directory layout described in PROBLEM.md / README.

CMake is fine. Any compiler/toolchain is fine. If your build targets Ubuntu 22.04 + x86-64-v3 and produces a working mlsys binary, you're good. 

A couple of practical notes so you don't get surprised at grading time: 
1. Static-link or bundle anything exotic. If your binary dynamically links against a library that isn't in a stock Ubuntu 22.04 install (e.g., a newer libstdc++ than GCC 11 ships, MKL, CUDA runtime, etc.), include it in your submission or statically link it. The grading environment is a clean Ubuntu 22.04 image. 
2. No GPU / no network at grading time. CPU-only, offline.
3. Smoke-test in a clean container before submitting. The one-liner from #72 is just an easy way to do that — feel free to adapt it to your CMake flow 
</comment>

---

## Comment 2
- author: yarongmu-google
- created_at: 2026-04-22T19:37:15Z
- url: https://github.com/yarongmu-google/MLSys/issues/83#issuecomment-4299378770

<comment>
I  will resolve this for now. Please open a new issue, referencing this one, if the above doesn't make sense.
</comment>

---

