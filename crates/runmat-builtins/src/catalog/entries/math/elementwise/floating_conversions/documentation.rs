use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const DOUBLE_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/double.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Host conversion and representation tests",
            location: "builtins::math::elementwise::double::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider conversion and prototype tests",
            location: "builtins::math::elementwise::double::tests::double_gpu_roundtrip",
        },
    ],
    notes: &[],
};

const DOUBLE_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`Y = double(X)` converts supported values to IEEE binary64 while preserving shape. Real and complex numeric input retains its real or complex domain; logical values become zero or one; characters become Unicode code points.",
        "All eight fixed-width integer classes convert from authoritative native storage. Binary64 cannot represent every `int64` or `uint64` value exactly, so values outside its consecutive-integer range may round.",
        "String scalars and arrays are parsed elementwise after trimming whitespace; text that is not one numeric value produces NaN. Sparse input retains CSC structure with double stored values. Unsupported cells, structs, objects, handles, and foreign values return a typed conversion error.",
        "`double(X, \"like\", prototype)` is a RunMat extension that keeps the output class fixed as double while selecting host or compatible provider residency from the prototype.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "A provider with true binary64 conversion support keeps real or complex input on its owning device. The result is a new double-precision handle with the original shape.",
        "If the owner cannot produce binary64, RunMat gathers through that exact owner and returns authoritative host storage. A GPU `\"like\"` prototype requires a live compatible owner; RunMat does not silently substitute a different provider or a lower-precision buffer.",
    ] },
];

const DOUBLE_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "integer", title: "Convert integers to double precision", program: "ints = int32([1 2 3]);\ndoubles = double(ints)", display_output: Some("doubles = [1 2 3]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(doubles, \"double\"));\nassert(isequal(doubles, [1 2 3]));" } },
    BuiltinExample { id: "logical", title: "Promote a logical mask for arithmetic", program: "mask = logical([0 1 0 1]);\nweights = double(mask)", display_output: Some("weights = [0 1 0 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(weights, \"double\"));\nassert(isequal(weights, [0 1 0 1]));" } },
    BuiltinExample { id: "characters", title: "Convert characters to Unicode code points", program: "codes = double('RunMat')", display_output: Some("codes = [82 117 110 77 97 116]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(codes, \"double\"));\nassert(isequal(codes, [82 117 110 77 97 116]));" } },
    BuiltinExample { id: "complex", title: "Preserve complex values while promoting precision", program: "z = [1+2i 3-4i];\nresult = double(z)", display_output: Some("result = [1+2i 3-4i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(result, \"double\"));\nassert(max(abs(result - z)) < 1e-12);" } },
    BuiltinExample { id: "gpu-array", title: "Convert provider-resident single data to double", program: "G = single(gpuArray([1 2; 3 4]));\nH = double(G);\nhost = gather(H)", display_output: Some("host = [1 2; 3 4]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(host, \"double\"));\nassert(isequal(host, [1 2; 3 4]));" } },
    BuiltinExample { id: "sparse", title: "Preserve sparse structure while converting values", program: "S = sparse([1 3], [1 2], [4 -1], 3, 2);\nA = double(S)", display_output: Some("A is a 3-by-2 double sparse matrix with two stored values"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(issparse(A));\nassert(isa(A, \"double\"));\nassert(nnz(A) == 2);\nassert(isequal(full(A), [4 0; 0 0; 0 -1]));" } },
    BuiltinExample { id: "like-extension", title: "Select host output residency with a prototype", program: "prototype = single(0);\nout = double([pi 0], \"like\", prototype)", display_output: Some("out = [3.1416 0]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(out, \"double\"));\nassert(max(abs(out - [pi 0])) < 1e-12);" } },
    BuiltinExample { id: "matrix-shape", title: "Promote a matrix without changing shape", program: "A = single([1.5 2.25; 3.75 4.5]);\nB = double(A)", display_output: Some("B = [1.5 2.25; 3.75 4.5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(B, \"double\"));\nassert(isequal(size(B), [2 2]));\nassert(isequal(B, [1.5 2.25; 3.75 4.5]));" } },
];

const DOUBLE_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `double` change values already stored as double?", answer: "No numeric conversion is required, although ordinary value ownership rules still apply." },
    BuiltinDocumentationFaq { question: "How are logical inputs handled?", answer: "Logical false and true become double zero and one." },
    BuiltinDocumentationFaq { question: "What happens to NaN or infinity?", answer: "IEEE NaN, positive and negative infinity, and signed zero are preserved." },
    BuiltinDocumentationFaq { question: "Can `double` convert strings?", answer: "Yes. Each string element is parsed as one numeric value; invalid text produces NaN. Character arrays instead become Unicode code points." },
    BuiltinDocumentationFaq { question: "What happens to sparse input?", answer: "CSC structure is retained and each stored value is converted to double." },
    BuiltinDocumentationFaq { question: "Will `double` keep results on the GPU?", answer: "Yes when the owning provider supports true binary64 output. Otherwise ordinary conversion gathers to host through the exact owner." },
    BuiltinDocumentationFaq { question: "Does `double` allocate new memory?", answer: "Conversion produces authoritative double storage. Fusion may eliminate a safe intermediate." },
    BuiltinDocumentationFaq { question: "Can I request GPU residency with `\"like\"`?", answer: "In RunMat mode, `double(X, \"like\", prototype)` selects compatible host or GPU residency while the output class remains double." },
    BuiltinDocumentationFaq { question: "How does `double` handle complex input?", answer: "Both components are retained and converted to double precision." },
];

pub(super) const DOUBLE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("double"), slug: Some("double"),
    summary: "Convert supported values to double-precision storage.",
    description: "`double(X)` converts supported scalars and arrays to IEEE binary64 while preserving shape and documented storage, domain, and residency behavior.",
    keywords: &["double", "float64", "binary64", "cast", "conversion", "gpuArray", "like"],
    related: &["single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "gather", "gpuArray"],
    sections: DOUBLE_SECTIONS, examples: DOUBLE_EXAMPLES, example_exemption: None, faqs: DOUBLE_FAQS,
    links: &[], media: &[], evidence: DOUBLE_EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

const SINGLE_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/single.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host conversion and representation tests", location: "builtins::math::elementwise::single::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider conversion and prototype tests", location: "builtins::math::elementwise::single::tests::single_gpu_roundtrip" },
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
