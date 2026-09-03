use super::support::define_binary_bitwise_entry;
use crate::{
    BinaryBitwiseOperator, BuiltinDocumentation, BuiltinDocumentationAuthority,
    BuiltinDocumentationEvidence, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

define_binary_bitwise_entry!(
    entry: BITXOR_CATALOG_ENTRY,
    descriptor: BITXOR_DESCRIPTOR,
    extensions: BITXOR_EXTENSIONS,
    single_extension: BITXOR_SINGLE_INPUT_EXTENSION,
    gpu_domain_extension: BITXOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    gpu_assumed_extension: BITXOR_GPU_ASSUMED_TYPE_EXTENSION,
    integer_capabilities: BITXOR_INTEGER_CAPABILITIES,
    name: "bitxor",
    upper: "Bitxor",
    operator: BinaryBitwiseOperator::Xor,
    documentation: BITXOR_DOCUMENTATION
);

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`C = bitxor(A, B)` applies bitwise exclusive OR to same-class logical or integer inputs and to finite nonnegative integer-valued `double` inputs. Compatible shapes use implicit expansion.",
        "Inputs must share a data type unless one operand is a scalar `double`. Logical inputs return logical output, fixed-width integer inputs preserve their native class, and two `double` inputs return `double`.",
        "The optional `assumedtype` interprets `double` inputs using a named signed or unsigned integer width. Typed integer inputs must match that class.",
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
        title: "Find differing bits",
        program: "c = bitxor(6, 3)",
        display_output: Some("c = 5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(c == 5); assert(isa(c, 'double'));",
        },
    },
    BuiltinExample {
        id: "typed",
        title: "Preserve a fixed-width class",
        program: "a = uint8([1 3 7]);\nc = bitxor(a, uint8(3))",
        display_output: Some("c = uint8([2 0 4])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(c, 'uint8')); assert(isequal(c, uint8([2 0 4])));",
        },
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "bitand",
        target: BuiltinDocumentationLinkTarget::Builtin("bitand"),
    },
    BuiltinDocumentationLink {
        label: "bitor",
        target: BuiltinDocumentationLinkTarget::Builtin("bitor"),
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
const BITXOR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bitxor"),
    slug: Some("bitxor"),
    summary: "Compute bitwise exclusive OR for integer-valued scalars and arrays.",
    description: "`bitxor` sets each result bit when the corresponding input bits differ.",
    keywords: &["bitxor", "bitwise", "xor", "integer", "gpuArray"],
    related: &["bitand", "bitor", "bitshift"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
