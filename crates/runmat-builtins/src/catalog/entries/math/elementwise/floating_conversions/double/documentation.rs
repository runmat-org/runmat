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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/floating_conversions/double/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Host conversion and representation tests",
            location: "builtins::math::elementwise::floating_conversions::double::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider conversion and prototype tests",
            location: "builtins::math::elementwise::floating_conversions::double::tests::provider::double_gpu_roundtrip",
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
    BuiltinExample { id: "integer", title: "Convert integers to double precision", program: "ints = int32([1 2 3]);\ndoubles = double(ints)", display_output: Some("doubles = [1 2 3]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(doubles, \"double\"));\nassert(isequal(doubles, [1 2 3]));" } },
    BuiltinExample { id: "logical", title: "Promote a logical mask for arithmetic", program: "mask = logical([0 1 0 1]);\nweights = double(mask)", display_output: Some("weights = [0 1 0 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(weights, \"double\"));\nassert(isequal(weights, [0 1 0 1]));" } },
    BuiltinExample { id: "characters", title: "Convert characters to Unicode code points", program: "codes = double('RunMat')", display_output: Some("codes = [82 117 110 77 97 116]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(codes, \"double\"));\nassert(isequal(codes, [82 117 110 77 97 116]));" } },
    BuiltinExample { id: "complex", title: "Preserve complex values while promoting precision", program: "z = [1+2i 3-4i];\nresult = double(z)", display_output: Some("result = [1+2i 3-4i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(result, \"double\"));\nassert(max(abs(result - z)) < 1e-12);" } },
    BuiltinExample { id: "gpu-array", title: "Convert provider-resident single data to double", program: "G = single(gpuArray([1 2; 3 4]));\nH = double(G);\nhost = gather(H)", display_output: Some("host = [1 2; 3 4]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(host, \"double\"));\nassert(isequal(host, [1 2; 3 4]));" } },
    BuiltinExample { id: "sparse", title: "Preserve sparse structure while converting values", program: "S = sparse([1 3], [1 2], [4 -1], 3, 2);\nA = double(S)", display_output: Some("A is a 3-by-2 double sparse matrix with two stored values"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(issparse(A));\nassert(isa(A, \"double\"));\nassert(nnz(A) == 2);\nassert(isequal(full(A), [4 0; 0 0; 0 -1]));" } },
    BuiltinExample { id: "like-extension", title: "Select host output residency with a prototype", program: "prototype = single(0);\nout = double([pi 0], \"like\", prototype)", display_output: Some("out = [3.1416 0]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(out, \"double\"));\nassert(max(abs(out - [pi 0])) < 1e-12);" } },
    BuiltinExample { id: "matrix-shape", title: "Promote a matrix without changing shape", program: "A = single([1.5 2.25; 3.75 4.5]);\nB = double(A)", display_output: Some("B = [1.5 2.25; 3.75 4.5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(B, \"double\"));\nassert(isequal(size(B), [2 2]));\nassert(isequal(B, [1.5 2.25; 3.75 4.5]));" } },
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
