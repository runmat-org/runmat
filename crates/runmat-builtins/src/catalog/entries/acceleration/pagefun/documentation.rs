use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("pagefun"), slug: Some("pagefun"),
 summary: "Apply matrix operators page-by-page across higher-dimensional arrays.", description: "`pagefun(func, A, B, ...)` applies supported operators independently across pages indexed by third-and-higher dimensions, following MATLAB-compatible pagewise execution semantics.",
 keywords: &["pagefun",
"gpuArray",
"batched matrix multiply",
"mtimes",
"pages"], related: &["mtimes","gpuArray"],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"MATLAB-compatible calls require a function handle. In `runmat` compatibility mode, builtin names supplied as character vectors or string scalars are accepted as a RunMat extension.",
"The first two dimensions of each input form the matrix operated on by the builtin. Dimensions three and up identify individual pages. Inputs with fewer pages are broadcast so long as the page extent is `1`.",
"Any combination of empty matrices and empty page extents produces empty results with MATLAB-compatible shapes.",
"MATLAB-compatible calls require at least one distributed or `gpuArray` input. In `runmat` compatibility mode, all-host operands are accepted as a RunMat extension and return host results.",
"When any operand is a `gpuArray`, provider fallback gathers inputs through the provider that owns them and restores supported results to that same provider. Automatic residency remains transparent, while an explicit `gpuArray` remains resident whenever its class can be represented by the owner.",
"Integer operands inherit the selected function's contract. For `@mtimes`, one operand must be scalar when either operand is an integer class; each page uses exact native integer storage, applies the class-preserving saturation rules of `mtimes`, and never passes through a floating copy."
] },
BuiltinDocumentationSection { heading: "GPU residency", paragraphs: &[
"WGPU provider accelerates batched @mtimes; if no provider hook is present the runtime evaluates on the host and re-uploads numeric results.",
"RunMat Accelerate exposes a custom `pagefun` provider hook. When every operand is a floating `gpuArray` and the owning provider implements the operation (the WGPU backend does for `@mtimes`), pages stay on the device and execute via a tiled compute shader. Mixed host/GPU inputs, integer pages, and unsupported provider operations use owner-aware host fallback; exact native storage is retained and supported outputs are restored to the original provider.",
"No. RunMat's auto-offload planner migrates tensors to the GPU when beneficial. When you explicitly pass any `gpuArray` input, `pagefun` returns supported real or complex numeric results on the device whenever the provider can accept the upload."
] },
BuiltinDocumentationSection { heading: "Additional usage", paragraphs: &[
"Batched matrix multiplication across pages\nA = reshape(1:12, [2 2 3]);\nB = reshape(13:24, [2 2 3]);\nC = pagefun(@mtimes, A, B)\nExpected display: C(:,:,1) =\n    55    63\n    82    94\n\nC(:,:,2) =\n   211   235\n   246   274\n\nC(:,:,3) =\n   431   471\n   474   518",
"Broadcasting a single page against multiple pages\nA = reshape(1:8, [2 2 2]);\nB = eye(2);\nC = pagefun(@mtimes, A, B);   % B broadcasts across the page dimension",
"Working with `gpuArray` inputs\nG1 = gpuArray(rand(4, 4, 8));\nG2 = gpuArray(rand(4, 4, 8));\nH = pagefun(@mtimes, G1, G2);\nfirstPage = gather(H(:, :, 1));   % Inspect the first product page on the host",
"Handling empty page dimensions\nZ = zeros(2, 2, 0);\nR = pagefun(@mtimes, Z, Z);\nsize(R)\nExpected display: ans =\n     2     2     0"
] },
BuiltinDocumentationSection { heading: "Execution boundaries", paragraphs: &[
"Only the matrix-product operation is supported; arbitrary closures, user overrides and bound receivers are not executed. Unknown selectors can fail but do not introduce unknown callback effects.",
"Numeric arrays are the complete input and output surface. Nested cells, structures, foreign values and distributed arrays are rejected before provider work, so there are no nested resident outputs. Direct distributed input is unsupported; ordinary worker calls require their own provider ownership and the shared runtime kernel.",
"Native/JIT/AOT and browser execution use the portable typed runtime kernel. Workload features are matrix extents, page broadcast extents, classes, complexity, requested output count and resident transfer requirements. Shared placement services own timing, scheduling and feedback; no per-builtin thresholds are used.",
"Provider outputs must match owner, device, shape, precision, real storage and class and must not alias inputs. Invalid outputs are released without freeing borrowed inputs. Only typed unsupported-operation failures allow host fallback; other provider failures remain visible.",
"The single result may be discarded, but admission, shape checks and computation still occur. More than one output is rejected before provider access."
] },
BuiltinDocumentationSection { heading: "Documentation reconciliation", paragraphs: &[
"Legacy signatures, summary, description, behavior, examples, FAQs, source and links are preserved or corrected here. Fusion flags and f32/f64 provider support are canonical placement/operation facts; no feature is required. Old display text is explanatory, not a numerical test oracle."
] }],
 examples: &[BuiltinExample { id: "batch", title: "batch", program: "A = reshape(1:8,[2 2 2]); B = eye(2); C = pagefun(@mtimes,A,B);", display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(C,A));" } },
BuiltinExample { id: "empty", title: "empty", program: "Z = zeros(2,2,0); R = pagefun(@mtimes,Z,Z);", display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(R),[2 2 0]));" } },
BuiltinExample { id: "integer", title: "integer", program: "A = intmax('uint64'); R = pagefun(@mtimes,A,uint64(1));", display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isa(R,'uint64')); assert(R == A);" } },
BuiltinExample { id: "resident", title: "resident", program: "A = single(reshape(1:8,[2 2 2])); G = gpuArray(A); I = gpuArray(single(eye(2))); H = pagefun(@mtimes,G,I); C = gather(H);", display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isgpuarray(H)); assert(isa(C,'single')); assert(isequal(C,A));" } }], example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Which functions does `pagefun` support today?", answer: "RunMat currently supports `@mtimes` page-wise. Additional MATLAB page-aware functions will be added over time as the GPU provider hooks land." },
BuiltinDocumentationFaq { question: "How are page dimensions inferred?", answer: "The first two dimensions represent the matrix operated on by the builtin. Any remaining dimensions are treated as page indices. Inputs with fewer trailing dimensions receive implicit singleton expansion to match other operands." },
BuiltinDocumentationFaq { question: "What happens if the pages are incompatible?", answer: "`pagefun` raises a MATLAB-compatible error describing the mismatched page dimension. Matrix dimension mismatches are forwarded from the underlying builtin (for example, `Inner matrix dimensions must agree` for `@mtimes`)." },
BuiltinDocumentationFaq { question: "Are results always uploaded back to the GPU?", answer: "When any operand is a `gpuArray` and an acceleration provider is registered, supported uniform real or complex numeric results are uploaded back to the device. An all-host RunMat-extension call returns a host result." },
BuiltinDocumentationFaq { question: "Does `pagefun` participate in fusion?", answer: "No. Because `pagefun` performs independent page products it forms a fusion barrier. Upstream expressions are evaluated before entering `pagefun`." }],
 links: &[BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") },
BuiltinDocumentationLink { label: "arrayfun", target: BuiltinDocumentationLinkTarget::Documentation("./arrayfun") },
BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Documentation("./gather") },
BuiltinDocumentationLink { label: "gpuDevice", target: BuiltinDocumentationLinkTarget::Documentation("./gpudevice") },
BuiltinDocumentationLink { label: "gpuInfo", target: BuiltinDocumentationLinkTarget::Documentation("./gpuinfo") }], media: &[],
 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/acceleration/gpu/pagefun.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Page semantics and provider regressions", location: "builtins::acceleration::gpu::pagefun::tests" }], notes: &[] },
 introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
