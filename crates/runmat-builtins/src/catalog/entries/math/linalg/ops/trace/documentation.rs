use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`trace(A)` returns the sum of the main diagonal of a square full or sparse single- or double-precision matrix.",
            "The input must be square and two-dimensional. Rectangular matrices, nonscalar vectors, and arrays with nonsingleton trailing dimensions raise an error.",
            "Scalars (real or complex) return their own value.",
            "Single input returns single, double input returns double, and complex floating input retains both its real and imaginary values and floating precision.",
            "RunMat mode additionally accepts logical, character, and native integer matrices. Integer diagonal values are accumulated exactly and converted to double only when the sum is exactly representable.",
            "Full and sparse real or complex floating matrices are supported. Sparse inputs are read directly from their typed CSC storage.",
            "Empty matrices yield `0`. Empty complex matrices yield `0 + 0i`.",
            "`gpuArray` inputs stay resident when the provider supports diagonal extraction and double accumulation. Known-unavailable routes are selected before staging and compute on the host, restoring a class-preserving 1×1 GPU result."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["When the input's owning provider implements diagonal extraction and reduction, `trace` remains resident. Otherwise it gathers once, computes on the host, and restores a 1×1 result with the original floating class through that same provider. Automatic residency and explicit `gpuArray` inputs use the same owner-preserving path.",
            "1. If the owning provider implements `diag_extract` and `reduce_sum`, the reduction stays resident. 2. Otherwise the matrix gathers once and a class-preserving 1×1 result is restored through that owner. 3. Integer, logical, and character inputs follow their compatibility-mode admission rules before provider work begins.",
            "Resident execution requires diagonal extraction and a double-accumulating reduction. If the provider certifies that route unavailable before staging, RunMat gathers once, computes on the host, and restores a class-preserving 1×1 result through the same owner. Failures after provider work starts are reported without replay."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "example-1", title: "Summing the diagonal of a square matrix", program: "A = [1 2 3; 4 5 6; 7 8 9];\nt = trace(A)", display_output: Some("t = 15"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(t == 15);" } },
    BuiltinExample { id: "example-2", title: "Rejecting a rectangular matrix", program: "B = [4 2; 1 3; 5 6];\nresult = trace(B)", display_output: Some("Error: trace requires a square matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:trace:InvalidInput" } },
    BuiltinExample { id: "example-3", title: "Getting the trace of a triangular matrix", program: "U = [4 1 2; 0 5 3; 0 0 6];\ntri_trace = trace(U)", display_output: Some("tri_trace = 15"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(tri_trace == 15);" } },
    BuiltinExample { id: "example-4", title: "Working with complex-valued matrices", program: "Z = [1+2i 2; 3 4-5i];\nzTrace = trace(Z)", display_output: Some("zTrace = 5.0000 - 3.0000i"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(zTrace == 5 - 3i);" } },
    BuiltinExample { id: "example-5", title: "Tracing a gpuArray and gathering the scalar result", program: "G = gpuArray(rand(1024));\ngpuResult = trace(G);     % gpuArray result; execution depends on the provider\nscalarHost = gather(gpuResult)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isgpuarray(gpuResult)); assert(isfinite(scalarHost));" } },
    BuiltinExample { id: "example-6", title: "Handling empty matrices safely", program: "E = zeros(0, 0);\nvalue = trace(E)", display_output: Some("value = 0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(value == 0);" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What happens if my matrix is not square?", answer: "It raises `RunMat:trace:InvalidInput`; `trace` requires a square matrix." },
    BuiltinDocumentationFaq { question: "Does `trace` accept higher-dimensional arrays?", answer: "Only when trailing dimensions are singleton. Otherwise it raises an error because MATLAB restricts `trace` to 2-D matrix slices." },
    BuiltinDocumentationFaq { question: "How are logical inputs handled?", answer: "Logical matrices are a RunMat-mode extension and return a double sum. MATLAB compatibility modes accept the documented single and double matrix classes." },
    BuiltinDocumentationFaq { question: "What is returned for empty inputs?", answer: "Empty real matrices produce `0`; empty complex matrices produce `0 + 0i`, exactly like MATLAB." },
    BuiltinDocumentationFaq { question: "Does the result stay on the GPU?", answer: "The result remains a `gpuArray`. Execution stays resident when the provider supports the required double accumulation; a known-unavailable route gathers before staging and restores the scalar afterward. A provider failure after work starts is reported without replay." },
    BuiltinDocumentationFaq { question: "Can I call `trace` on complex data?", answer: "Yes. Full and sparse complex inputs return the sum of the diagonal's real and imaginary parts, preserving single or double precision." },
    BuiltinDocumentationFaq { question: "Is there any precision loss with large matrices?", answer: "Single and complex-single inputs retain single precision; double and complex-double inputs retain double precision. RunMat-mode integer inputs use exact wide accumulation and reject a final sum that cannot be represented exactly as double." },
    BuiltinDocumentationFaq { question: "Does `trace` modify the input matrix?", answer: "No. It reads the diagonal and returns a new scalar without altering the original matrix or its residency." },
    BuiltinDocumentationFaq { question: "How does `trace` interact with sparse matrices?", answer: "Real and complex sparse single and double matrices are supported directly. Native sparse integer matrices are available in RunMat mode and use the same exact-sum boundary as dense integer matrices." },
    BuiltinDocumentationFaq { question: "Can I rely on `trace` inside fused GPU expressions?", answer: "Fused kernels treat `trace` as a scalar reduction boundary. The planner emits GPU kernels when hooks are available; otherwise it falls back gracefully." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "diag", target: BuiltinDocumentationLinkTarget::Builtin("diag") },
    BuiltinDocumentationLink { label: "sum", target: BuiltinDocumentationLinkTarget::Builtin("sum") },
    BuiltinDocumentationLink { label: "mtimes", target: BuiltinDocumentationLinkTarget::Builtin("mtimes") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "ctranspose", target: BuiltinDocumentationLinkTarget::Builtin("ctranspose") },
    BuiltinDocumentationLink { label: "dot", target: BuiltinDocumentationLinkTarget::Builtin("dot") },
    BuiltinDocumentationLink { label: "mldivide", target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },
    BuiltinDocumentationLink { label: "mpower", target: BuiltinDocumentationLinkTarget::Builtin("mpower") },
    BuiltinDocumentationLink { label: "mrdivide", target: BuiltinDocumentationLinkTarget::Builtin("mrdivide") },
    BuiltinDocumentationLink { label: "transpose", target: BuiltinDocumentationLinkTarget::Builtin("transpose") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/trace.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("trace"),
    slug: Some("trace"),
    summary: "Sum the main-diagonal elements of a square matrix.",
    description: "`trace(A)` returns the sum of the main diagonal of a square full or sparse single- or double-precision matrix.",
    keywords: &["trace", "matrix trace", "diagonal sum", "gpu"],
    related: &["diag", "sum", "mtimes", "gpuarray", "gather", "ctranspose", "dot", "mldivide", "mpower", "mrdivide", "transpose"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/trace.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::ops::trace::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::ops::trace::tests::trace_gpu_provider_roundtrip" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Existing GPU parity tests", location: "builtins::math::linalg::ops::trace::tests::trace_wgpu_matches_cpu" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
