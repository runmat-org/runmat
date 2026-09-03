use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = swapbytes(X)` reverses the bytes within each element of a real numeric scalar or dense array. The output retains the input class and shape. Eight-bit integer elements are unchanged because each contains one byte.",
            "All eight fixed-width integer classes are transformed in their native storage. `double` and `single` values are transformed through their exact IEEE bit patterns, including signed zero, infinities, and NaN payloads; the implementation does not convert `single` through `double`.",
            "Logical, complex, sparse, character, string, cell, struct, and object inputs are outside the supported input domain and produce a structured error.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &[
            "`swapbytes` is a host byte-layout operation. Automatically resident input gathers to the host before the byte reversal. Explicit `gpuArray` input uses the same exact host implementation only in RunMat compatibility mode; MATLAB compatibility mode rejects that RunMat extension before provider access.",
            "The result is host-resident. Use `gpuArray` after `swapbytes` when a subsequent operation should run on a provider.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "uint16-scalar",
        title: "Reverse a two-byte unsigned integer",
        program: "x = uint16(hex2dec('1234'));\ny = swapbytes(x)",
        display_output: Some("y = uint16(13330), whose hexadecimal form is 3412"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, 'uint16')); assert(y == uint16(hex2dec('3412')));",
        },
    },
    BuiltinExample {
        id: "uint16-array",
        title: "Reverse every element of an array",
        program: "x = uint16([hex2dec('1234') hex2dec('00ff')]);\ny = swapbytes(x)",
        display_output: Some("y = uint16([hex2dec('3412') hex2dec('ff00')])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = uint16([hex2dec('3412') hex2dec('ff00')]); assert(isa(y, 'uint16')); assert(isequal(y, expected));",
        },
    },
    BuiltinExample {
        id: "uint8-identity",
        title: "Keep one-byte elements unchanged",
        program: "x = uint8([0 1 127 255]);\ny = swapbytes(x)",
        display_output: Some("y = uint8([0 1 127 255])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, 'uint8')); assert(isequal(y, x));",
        },
    },
    BuiltinExample {
        id: "signed-int32",
        title: "Preserve a signed integer class",
        program: "x = int32(hex2dec('01020304'));\ny = swapbytes(x)",
        display_output: Some("y = int32(hex2dec('04030201'))"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, 'int32')); assert(y == int32(hex2dec('04030201')));",
        },
    },
    BuiltinExample {
        id: "shape",
        title: "Preserve an array's shape",
        program: "x = reshape(uint16([1 2 3 4 5 6]), [2 3]);\ny = swapbytes(x)",
        display_output: Some("y is a 2-by-3 uint16 array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, 'uint16')); assert(isequal(size(y), [2 3])); assert(isequal(swapbytes(y), x));",
        },
    },
    BuiltinExample {
        id: "explicit-gpu-fallback",
        title: "Use the explicit host fallback in RunMat mode",
        program: "x = gpuArray(uint16([hex2dec('1234') hex2dec('00ff')]));\ny = swapbytes(x)",
        display_output: Some("y is a host uint16 array with reversed bytes"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = uint16([hex2dec('3412') hex2dec('ff00')]); assert(isa(y, 'uint16')); assert(isequal(y, expected));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `swapbytes` preserve integer class?", answer: "Yes. Every fixed-width integer class is transformed in its native storage and returned with the same class and shape. Eight-bit values are unchanged." },
    BuiltinDocumentationFaq { question: "Does floating-point byte swapping perform a numeric conversion?", answer: "No. RunMat reverses the bytes of each value's `double` or `single` bit pattern and reconstructs the same floating class." },
    BuiltinDocumentationFaq { question: "What happens with resident input?", answer: "Automatically resident input gathers transparently. Explicit `gpuArray` input may use the host fallback in RunMat compatibility mode. The result is host-resident." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "bitget", target: BuiltinDocumentationLinkTarget::Builtin("bitget") },
    BuiltinDocumentationLink { label: "bitset", target: BuiltinDocumentationLinkTarget::Builtin("bitset") },
    BuiltinDocumentationLink { label: "bitshift", target: BuiltinDocumentationLinkTarget::Builtin("bitshift") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/swapbytes.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Byte-swap runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/swapbytes.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "All integer widths, floating bit patterns, shape, and invalid inputs", location: "crates/runmat-runtime/src/builtins/math/bitwise/swapbytes/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Automatic gather and explicit compatibility gate", location: "crates/runmat-runtime/src/builtins/math/bitwise/swapbytes/tests.rs" },
    ],
    notes: &[],
};

pub(crate) const SWAPBYTES_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("swapbytes"),
    slug: Some("swapbytes"),
    summary: "Reverse the byte order within each numeric element.",
    description: "`swapbytes` reverses each element's native byte sequence while retaining numeric class and shape.",
    keywords: &["swapbytes", "byte order", "endianness", "integer", "single", "double"],
    related: &["bitget", "bitset", "bitshift", "gpuArray"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
