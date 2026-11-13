use super::support::define_binary_bitwise_entry;
use crate::{
    BinaryBitwiseOperator, BuiltinDocumentation, BuiltinDocumentationAuthority,
    BuiltinDocumentationEvidence, BuiltinDocumentationFaq, BuiltinDocumentationLink,
    BuiltinDocumentationLinkTarget, BuiltinDocumentationSection, BuiltinDocumentationStatus,
    BuiltinEvidenceKind, BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility,
    BuiltinExampleHarness, BuiltinExampleVerification,
};

define_binary_bitwise_entry!(
    entry: BITOR_CATALOG_ENTRY,
    descriptor: BITOR_DESCRIPTOR,
    extensions: BITOR_EXTENSIONS,
    single_extension: BITOR_SINGLE_INPUT_EXTENSION,
    gpu_domain_extension: BITOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    gpu_assumed_extension: BITOR_GPU_ASSUMED_TYPE_EXTENSION,
    integer_capabilities: BITOR_INTEGER_CAPABILITIES,
    name: "bitor",
    upper: "Bitor",
    operator: BinaryBitwiseOperator::Or,
    documentation: BITOR_DOCUMENTATION
);

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`C = bitor(A, B)` applies bitwise OR to same-class logical or integer inputs and to finite nonnegative integer-valued `double` inputs. Compatible shapes use implicit expansion.",
        "Inputs must share a data type unless one operand is a scalar `double`. Logical inputs return logical output, fixed-width integer inputs preserve their native class, and two `double` inputs return `double`.",
        "`C = bitor(A, B, assumedtype)` interprets `double` inputs using the named signed or unsigned integer width. Typed integer inputs must match `assumedtype`. Signed values use their two's-complement bit patterns.",
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
        program: "c = bitor(5, 3)",
        display_output: Some("c = 7"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(c == 7); assert(isa(c, 'double'));",
        },
    },
    BuiltinExample {
        id: "typed-broadcast",
        title: "Broadcast a typed scalar mask",
        program: "a = uint16([1 2 4]);\nc = bitor(a, uint16(8))",
        display_output: Some("c = uint16([9 10 12])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(c, 'uint16')); assert(isequal(c, uint16([9 10 12])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[BuiltinDocumentationFaq {
    question: "Can `bitor` broadcast a scalar mask?",
    answer: "Yes. A scalar expands across a compatible array shape.",
}];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "bitand",
        target: BuiltinDocumentationLinkTarget::Builtin("bitand"),
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
    implementation: &[BuiltinDocumentationLink {
        label: "Binary bitwise runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/bitwise/binary",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Typed, logical, sparse, broadcast, compatibility, and resident behavior",
        location: "crates/runmat-runtime/src/builtins/math/bitwise/engine/tests",
    }],
    notes: &[],
};
const BITOR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bitor"),
    slug: Some("bitor"),
    summary: "Compute bitwise OR for integer-valued scalars and arrays.",
    description:
        "`bitor` combines corresponding bits in compatible logical or integer-valued inputs.",
    keywords: &["bitor", "bitwise", "or", "integer", "gpuArray"],
    related: &["bitand", "bitxor", "bitshift"],
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
