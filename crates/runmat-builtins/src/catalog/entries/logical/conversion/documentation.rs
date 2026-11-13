use crate::*;

const SECTIONS: &[
BuiltinDocumentationSection] = &[

BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"`logical` accepts scalars, dense arrays, N-D tensors, and gpuArrays. Shapes are preserved bit-for-bit.",
"Finite non-zero real numeric values and real infinities map to `true`; `0` and `-0` map to `false`.",
"NaN and complex numeric inputs raise conversion errors rather than being coerced.",
"Character arrays are converted elementwise by interpreting code points (so `'A'` becomes `true`, `'\\0'` becomes `false`).",
"When RunMat extensions are enabled, string arrays convert elementwise by whether each element is nonempty; scalar strings remain unsupported.",
"Structs, cells, objects, and other unsupported types raise conversion errors.",
"Scalar results become logical scalars (`true`/`false`); dense higher-rank arrays produce dense logical arrays, and sparse inputs preserve sparse storage."] }
,

BuiltinDocumentationSection {
 heading: "GPU behavior",
 paragraphs: &[
"When the acceleration provider supports the required hooks, RunMat creates a logical result on the input's exact GPU owner. If fallback is needed, automatically resident inputs may return a host logical value; explicitly resident inputs are restored to the originating owner or the operation fails rather than silently changing the residency contract.",
"When a GPU provider implements `elem_ne` and `zeros_like`, RunMat evaluates `elem_ne(X, 0)` into a new device result and marks that result as logical. The result is validated before use and does not alias the input.",
"If the provider cannot service the request, RunMat converts through the host. Automatically resident inputs may remain on the host; explicit `gpuArray` inputs are re-uploaded to the same owner or return an error.",
"Handles that are already flagged as logical (`gpuArray.logical`) are returned without modification.",
"Scalars remain scalars: converting a `gpuArray` scalar preserves the residency and returns a logical gpuArray scalar.",
"Prefers a device-side elem_ne(X, 0) cast when the provider supports elem_ne and zeros_like. Provider results are validated for exact owner, shape, storage kind, and aliasing."] }
,

BuiltinDocumentationSection {
 heading: "Fusion",
 paragraphs: &[
"Logical conversion remains a fusion boundary; no elementwise template is currently registered."] }
,

BuiltinDocumentationSection {
 heading: "Integer and execution contract",
 paragraphs: &[
"All eight fixed-width integer classes convert without binary64 materialization. One input and at most one output are supported; discarded outputs still perform validation. NaN, complex, and unsupported inputs raise RunMat:logical:ConversionNotPossible. String arrays and symbolic constants require their RunMat extension gates.",
"Execution uses a portable asynchronous runtime binding with no workspace, filesystem, UI, callback, or random-service access. Distributed inputs are materialized by the shared argument boundary; foreign values are rejected. Provider allocation, synchronization, transfer, and fallback costs remain owned by the shared placement layer."] }
,

];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",

        title: "Creating a logical mask from numeric data",

        program: "values = [0 2 -3 0];\nmask = logical(values)",

        display_output: Some("mask =\n  1×4 logical array\n     0     1     1     0"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(mask, [false true true false]));",
        },
    },
    BuiltinExample {
        id: "example-2",

        title: "Building a logical mask from a matrix",

        program: "M = [-4 0 8; 0 1 0];\nmask = logical(M)",

        display_output: Some("mask =\n  2×3 logical array\n     1     0     1\n     0     1     0"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(mask, [true false true; false true false]));",
        },
    },
    BuiltinExample {
        id: "example-3",

        title: "Treating real infinities as true",

        program: "flags = logical([-Inf Inf 0])",

        display_output: Some("flags =\n  1×3 logical array\n     1     1     0"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(flags, [true true false]));",
        },
    },
    BuiltinExample {
        id: "example-4",

        title: "Rejecting complex numeric conversion",

        program: "logical(3 + 4i)",

        display_output: Some(
            "Error: conversion to logical from complex numeric data is not possible",
        ),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "MATLAB:logical:ConversionNotPossible",
        },
    },
    BuiltinExample {
        id: "example-5",

        title: "Converting character arrays to logical values",

        program: "chars = ['A' 0 'C'];\nmask = logical(chars)",

        display_output: Some("mask =\n  1×3 logical array\n     1     0     1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(mask, [true false true]));",
        },
    },
    BuiltinExample {
        id: "example-6",

        title: "Keeping gpuArray inputs on the device",

        program: "G = gpuArray([0 1 2]);\nmaskGPU = logical(G);\nhostMask = gather(maskGPU)",

        display_output: Some("hostMask =\n  1×3 logical array\n     0     1     1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Wgpu,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isgpuarray(maskGPU)); assert(isequal(hostMask, [false true true]));",
        },
    },
    BuiltinExample {
        id: "example-7",

        title: "Preserving empty shapes through logical conversion",

        program: "emptyVec = zeros(0, 3);\nlogicalEmpty = logical(emptyVec)",

        display_output: Some("logicalEmpty =\n  0×3 logical array\n     []"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(islogical(logicalEmpty)); assert(isequal(size(logicalEmpty), [0 3]));",
        },
    },
];
const FAQS: &[
BuiltinDocumentationFaq] = &[

BuiltinDocumentationFaq {
 question: "Which input types does `logical` support?",
 answer: "Real numeric, logical, character, sparse numeric, and supported gpuArray values are accepted. NaN and complex numeric values are rejected. Enabling RunMat extensions additionally accepts string arrays and symbolic numeric constants; structs, cells, objects, and function handles remain unsupported." }
,

BuiltinDocumentationFaq {
 question: "How are NaN or Inf values treated?",
 answer: "Real positive and negative infinity evaluate to `true`. NaN is rejected with a conversion error." }
,

BuiltinDocumentationFaq {
 question: "How does `logical` handle complex numbers?",
 answer: "Complex numeric inputs are rejected, including values whose imaginary component is zero." }
,

BuiltinDocumentationFaq {
 question: "Does the builtin change array shapes?",
 answer: "No. Shapes are preserved exactly, including empty dimensions and higher-rank tensors." }
,

BuiltinDocumentationFaq {
 question: "What happens to existing logical arrays?",
 answer: "They are returned verbatim. Logical gpuArrays remain on the device without triggering new allocations." }
,

BuiltinDocumentationFaq {
 question: "Can I convert strings with `logical`?",
 answer: "Scalar strings are rejected. When RunMat extensions are enabled, string arrays are accepted as an extension: each nonempty element becomes `true` and each empty element becomes `false`." }
,

BuiltinDocumentationFaq {
 question: "What about structs, cells, or objects?",
 answer: "They raise the same conversion error as MATLAB. Use functions like `~cellfun(@isempty, ...)` to derive masks instead." }
,

BuiltinDocumentationFaq {
 question: "Does the GPU path allocate new buffers?",
 answer: "Yes. The preferred device path computes `elem_ne` against a zero tensor into a distinct logical result. Fallback may create a host result, and explicit residency may require a new upload to the original owner." }
,

BuiltinDocumentationFaq {
 question: "Where can I learn more?",
 answer: "See the references below and the RunMat source for implementation details." }
,

];
const LINKS: &[
BuiltinDocumentationLink] = &[

BuiltinDocumentationLink {
 label: "`islogical`",
 target: BuiltinDocumentationLinkTarget::Builtin("islogical") }
,

BuiltinDocumentationLink {
 label: "`gpuArray`",
 target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") }
,

BuiltinDocumentationLink {
 label: "`gather`",
 target: BuiltinDocumentationLinkTarget::Builtin("gather") }
,

BuiltinDocumentationLink {
 label: "`find`",
 target: BuiltinDocumentationLinkTarget::Builtin("find") }
,

BuiltinDocumentationLink {
 label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/ops.rs") }
,

];
pub(super) const LOGICAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("logical"),
 slug: Some("logical"),
 summary: "Convert values to logical type in MATLAB and RunMat.",
 description: "`logical(X)` converts supported real numeric and character inputs into logical values. Zero maps to false and nonzero maps to true. NaN and complex numeric inputs are rejected. When RunMat extensions are enabled, string arrays and symbolic numeric constants are additionally accepted.",
 keywords: &[
"logical",
"boolean conversion",
"truth mask",
"gpuArray",
"mask array"],
 related: &[
"islogical",
 "gpuArray",
 "gather",
 "find"],
 sections: SECTIONS,
 examples: EXAMPLES,
 example_exemption: None,
 faqs: FAQS,
 links: LINKS,
 media: &[
],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[
],
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Exact host semantics, errors, extension gates, shape and provider behavior",
 location: "crates/runmat-runtime/src/builtins/logical/ops/tests.rs" }
,
 BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::WgpuTest,
 label: "Real-provider roundtrip (requires hardware)",
 location: "builtins::logical::ops::tests::logical_gpu_roundtrip" }
],
 notes: &[
"Legacy logical.json: title/category/keywords/summary/description/behaviors/FAQs/links/source retained; all seven examples gain executable value or error oracles. GPU/fusion/residency fields reconciled with the retained provider implementation and boundary contract. Feature requirement was null; no media were present."] }
,
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable) }
;
