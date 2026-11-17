# NOTES: kda_prefill_h8_d128_k_last

Process at a very high level as my personal notes, research ideas, etc are on paper. Happy to talk through them :)

# My process, at a high level:

1. I first pick an operation I think is worthwhile to focus on. I look at the model(s), and find a specific case on how it’s served including the exact kernel call. This decides the inputs, the outputs, and where the kernel starts and stops. I try to tag them with a source, assumption label, etc.
2. Then I write the definition as a slow, obviously correct reference first (naive pytorch), then the sizes (fixed vs changing), inputs and outputs, constraints, and a description of anything not covered simply from the shapes.
3. The hardest part is designing the tests. I go through the computation through my own systematic process to produce what I believe gives good çoverage.
4. I pass through the tests multiple times to make them more accurate and also actively red team them to gain a stronger test list. I also trim the tests down to what is neccesary (no duplication).
5. Then it just becomes doing a bunch of cross checking to ensure everything is working properly.
6. For this task specifically, I added the correct and broken implementations to show the grader in action.

# (Some) Design Decisions

1) I used sol-bench as I believe it to be a more explicit version of kernelbench as well as a more accurate benchmarking methodology for testing actual performance on hardware. Porting to kernelbench from sol-bench is trivial but untrue the opposite way.
2) I debated through how to define the problem through various implementations such as einsum, tensor comprehension comments, my own IRish format, but settled on the axes/inputs/outputs/constraints/description for coverage of the types of operations expressed in inference/training.
3) I ran into trouble defining my task as I underestimated the amount of encoding I needed to do for the task to be realisitc. I have a pretty good idea on what should be included now as a seperate task vs not and what to look for specifically.
4) How the operation is actually served actually required me to retroactively tone down some of my tests.
5) Extra: I noticed some interesting ways this operation could be implemented. kda chunked prefill can definetely be optimized faster than flashKDA's by using cuLA and this representation showed that slowing one kernel down can actually make the overall operation faster. Similarity to GDN prefill also exposed more neccesary tests that I have since added to my systematic process.

## Tolerances

Tolerances are calibrated on CPU against an fp32 chunked implementation (`solutions/correct_chunked.py`). They must be recalibrated against FlashKDA and FLA on GPU, because those use bf16 block math.
