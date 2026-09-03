use super::support::define_binary_bitwise_entry;
use crate::{
    BinaryBitwiseOperator, BuiltinDocumentation, BuiltinDocumentationAuthority,
    BuiltinDocumentationEvidence, BuiltinDocumentationFaq, BuiltinDocumentationLink,
    BuiltinDocumentationLinkTarget, BuiltinDocumentationSection, BuiltinDocumentationStatus,
    BuiltinEvidenceKind, BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility,
    BuiltinExampleHarness, BuiltinExampleVerification,
};

define_binary_bitwise_entry!(
    entry: BITAND_CATALOG_ENTRY,
    descriptor: BITAND_DESCRIPTOR,
    extensions: BITAND_EXTENSIONS,
    single_extension: BITAND_SINGLE_INPUT_EXTENSION,
    gpu_domain_extension: BITAND_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    gpu_assumed_extension: BITAND_GPU_ASSUMED_TYPE_EXTENSION,
    integer_capabilities: BITAND_INTEGER_CAPABILITIES,
    name: "bitand",
    upper: "Bitand",
    operator: BinaryBitwiseOperator::And,
    documentation: BITAND_DOCUMENTATION
);

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`C = bitand(A, B)` applies bitwise AND to same-class logical or integer inputs and to finite nonnegative integer-valued `double` inputs. Compatible shapes use implicit expansion.",
        "Inputs must share a data type unless one operand is a scalar `double`. Logical inputs return logical output, fixed-width integer inputs preserve their native class, and two `double` inputs return `double`.",
        "`C = bitand(A, B, assumedtype)` interprets `double` inputs using the named signed or unsigned integer width. Typed integer inputs must match `assumedtype`. Signed values use their two's-complement bit patterns.",
        "Fractional, out-of-range, infinite, NaN, complex, mixed integer-class, string, cell, struct, and incompatible-shape inputs produce structured errors.",
    ] },
    BuiltinDocumentationSection { heading: "Resident values", paragraphs: &[
        "Documented resident input uses `uint8`, `uint16`, or `uint32`. RunMat gathers those values exactly, evaluates the operation on the host, and restores the result to the owning provider.",
        "Single-precision input, broader resident classes, and resident calls with `assumedtype` are separately gated RunMat extensions.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Combine two scalar masks",
        program: "c = bitand(6, 3)",
        display_output: Some("c = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(c == 2); assert(isa(c, 'double'));",
        },
    },
    BuiltinExample {
        id: "typed-broadcast",
        title: "Broadcast a typed scalar mask",
        program: "a = uint16([3 7 15]);\nc = bitand(a, uint16(6))",
        display_output: Some("c = uint16([2 6 6])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(c, 'uint16')); assert(isequal(c, uint16([2 6 6])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[BuiltinDocumentationFaq {
    question: "Can `bitand` broadcast a scalar mask?",
    answer: "Yes. A scalar expands across a compatible array shape.",
}];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "bitor",
        target: BuiltinDocumentationLinkTarget::Builtin("bitor"),
    },
    BuiltinDocumentationLink {
        label: "bitxor",
        target: BuiltinDocumentationLinkTarget::Builtin("bitxor"),
    },
    BuiltinDocumentationLink {
        label: "bitshift",
        target: BuiltinDocumentationLinkTarget::Builtin("bitshift"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Binary bitwise runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/binary") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed, logical, sparse, broadcast, compatibility, and resident behavior", location: "crates/runmat-runtime/src/builtins/math/bitwise/binary/tests.rs" }],
    notes: &[],
};
const BITAND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bitand"),
    slug: Some("bitand"),
    summary: "Compute bitwise AND for integer-valued scalars and arrays.",
    description:
        "`bitand` combines corresponding bits in compatible logical or integer-valued inputs.",
    keywords: &["bitand", "bitwise", "and", "integer", "gpuArray"],
    related: &["bitor", "bitxor", "bitshift"],
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
