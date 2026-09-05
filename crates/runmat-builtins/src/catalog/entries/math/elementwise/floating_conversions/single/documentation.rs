use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SINGLE_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions/single/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host conversion and representation tests", location: "builtins::math::elementwise::floating_conversions::single::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider conversion and prototype tests", location: "builtins::math::elementwise::floating_conversions::single::tests::provider::single_gpu_roundtrip" },
    ],
    notes: &[],
};

const SINGLE_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`Y = single(X)` converts supported values to IEEE binary32 and preserves shape. Real and complex numeric input retains its domain; logical values become zero or one; characters become Unicode code points.",
        "All fixed-width integers convert directly from native integer storage to binary32. Values are rounded to the nearest representable single-precision value without an intermediate binary64 materialization.",
        "Sparse input retains CSC structure with single stored values. String, cell, struct, object, handle, and foreign input is rejected with a typed conversion error. Empty arrays retain their exact dimensions.",
        "`single(X, \"like\", prototype)` is a RunMat extension that keeps the output class fixed as single while selecting host or compatible provider residency from the prototype.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "Supported providers convert real input on its owning device and keep the result resident as native binary32. Complex conversion and unsupported provider operations use the typed owner-aware fallback.",
        "Fallback gathers through the exact owner, converts both complex components or real storage on the host, and re-uploads for an explicit GPU `\"like\"` prototype. Returned handles are validated for class, shape, owner, and non-aliasing.",
    ] },
];

const SINGLE_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrix", title: "Convert a matrix to single precision", program: "A = [1 2 3; 4 5 6];\nB = single(A)", display_output: Some("B is a 2-by-3 single matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(B, \"single\"));\nassert(isequal(size(B), [2 3]));\nassert(isequal(B, single([1 2 3; 4 5 6])));" } },
    BuiltinExample { id: "scalar", title: "Convert a scalar to single precision", program: "pi_single = single(pi)", display_output: Some("pi_single = single(3.1416)"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(pi_single, \"single\"));\nassert(abs(double(pi_single) - pi) < 2e-7);" } },
    BuiltinExample { id: "complex", title: "Convert complex numbers to single precision", program: "z = [1+2i 3-4i];\nsingle_z = single(z)", display_output: Some("single_z is a 1-by-2 complex single vector"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(single_z, \"single\"));\nassert(max(abs(double(single_z) - z)) < 1e-6);" } },
    BuiltinExample { id: "characters", title: "Convert characters to single-precision code points", program: "codes = single('ABC')", display_output: Some("codes = single([65 66 67])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(codes, \"single\"));\nassert(isequal(codes, single([65 66 67])));" } },
    BuiltinExample { id: "gpu-array", title: "Keep provider-resident input on the GPU", program: "G = gpuArray([0 3; 1 4; 2 5]);\nH = single(G);\nhost = gather(H)", display_output: Some("host is a 3-by-2 single matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(host, \"single\"));\nassert(isequal(host, single([0 3; 1 4; 2 5])));" } },
    BuiltinExample { id: "logical", title: "Convert a logical mask to single precision", program: "mask = logical([0 1 0 1]);\nweights = single(mask)", display_output: Some("weights = single([0 1 0 1])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(weights, \"single\"));\nassert(isequal(weights, single([0 1 0 1])));" } },
];

const SINGLE_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why can `single` change a numeric value?", answer: "Binary32 has about seven decimal digits of precision, so values are rounded to the nearest representable result." },
    BuiltinDocumentationFaq { question: "Does `single` accept integer classes?", answer: "Yes. Every fixed-width integer class converts directly from its native storage to binary32." },
    BuiltinDocumentationFaq { question: "Can `single` convert strings?", answer: "No. String input returns a typed conversion error; convert text intentionally before calling `single`." },
    BuiltinDocumentationFaq { question: "What about structs, cells, or user objects?", answer: "Unsupported container and object inputs return a conversion error. Extract the numeric value first." },
    BuiltinDocumentationFaq { question: "Does `single` support complex numbers?", answer: "Yes. Both real and imaginary components convert to native single precision." },
    BuiltinDocumentationFaq { question: "How does `single` handle empty arrays?", answer: "Empty arrays remain empty and retain all dimensions and orientation." },
    BuiltinDocumentationFaq { question: "Will GPU input stay on the GPU?", answer: "Supported providers cast on device. Typed fallback gathers through the owner and re-uploads when explicit output residency requires it." },
    BuiltinDocumentationFaq { question: "What do `class` and `isa` report?", answer: "Host and gathered results own native binary32 storage, so their class is `single`; this is not a label over a double buffer." },
    BuiltinDocumentationFaq { question: "Do NaN and infinity survive the cast?", answer: "Yes. IEEE NaN, positive and negative infinity, and signed zero are retained subject to binary32 precision." },
    BuiltinDocumentationFaq { question: "Can a prototype select output residency?", answer: "In RunMat mode, `single(X, \"like\", prototype)` selects compatible host or GPU residency while output class remains single." },
];

pub(super) const SINGLE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("single"), slug: Some("single"),
    summary: "Convert supported values to single-precision storage.",
    description: "`single(X)` converts supported scalars and arrays to native IEEE binary32 while preserving shape and documented storage, domain, and residency behavior.",
    keywords: &["single", "float32", "binary32", "cast", "conversion", "gpuArray", "like"],
    related: &["double", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "gather", "gpuArray"],
    sections: SINGLE_SECTIONS, examples: SINGLE_EXAMPLES, example_exemption: None, faqs: SINGLE_FAQS,
    links: &[], media: &[], evidence: SINGLE_EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
