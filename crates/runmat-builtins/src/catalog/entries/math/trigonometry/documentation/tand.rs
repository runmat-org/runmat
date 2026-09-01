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
            "`tand(X)` evaluates tangent with `X` measured in degrees. It returns exact zero at multiples of 180, exact positive or negative one at the corresponding multiples of 45, and signed infinity at odd multiples of 90. Other finite values use degree-scaled tangent; non-finite real input returns `NaN`.",
            "The operation is elementwise and preserves scalar, vector, matrix, empty, and N-D shape. Real and complex `single` input returns `single`; real and complex `double` input returns `double`. Complex input uses the analytic continuation after scaling by `pi/180`; canonical snapping applies only to real input.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat mode accepts all eight real fixed-width integer classes. Each native value is reduced exactly modulo 360 before entering floating evaluation, so `int64` and `uint64` values above `flintmax` retain exact canonical and pole behavior. The result is double.",
            "Logical and character arrays are separately gated RunMat extensions and return double. Character input is reduced from exact Unicode scalar values. Strings, sparse arrays, typed complex integers, tables, timetables, and tall containers are rejected; execution-owned distributed arrays use the catalog's unary mapping policy.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &["Provider-resident input gathers through its owner for canonical host evaluation and returns to that owner when class-preserving restoration succeeds. Fusion is disabled because `tan(X*pi/180)` would replace exact zeros, units, and signed-infinite poles with rounded approximations."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "canonical-angle",
        title: "Evaluate a canonical angle",
        program: "y = tand(45)",
        display_output: Some("y = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 1);",
        },
    },
    BuiltinExample {
        id: "signed-poles",
        title: "Return signed infinities at the poles",
        program: "y = tand([90 -90])",
        display_output: Some("y = [Inf -Inf]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [Inf -Inf]));",
        },
    },
    BuiltinExample {
        id: "exact-half-turn",
        title: "Return exact zero at a half-turn",
        program: "y = tand(180)",
        display_output: Some("y = 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 0);",
        },
    },
    BuiltinExample {
        id: "wide-integer-extension",
        title: "Reduce wide integers exactly",
        program: "x = 0xFFFFFFFFFFFFFFFFu64;\ny = tand(x)",
        display_output: Some("y equals tand(15) without converting x to double"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - tand(15)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "provider-restoration",
        title: "Restore canonical results to the provider",
        program: "G = gpuArray(single([0 45 90]));\ny = tand(G);\nhostY = gather(y)",
        display_output: Some("hostY is single [0 1 Inf]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, \"gpuArray\"));\nassert(isa(hostY, \"single\"));\nassert(isequal(hostY, single([0 1 Inf])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does tand(90) return Inf?",
        answer: "Degree tangent defines signed infinities at its odd-quarter-turn poles. RunMat recognizes those phases before general floating evaluation.",
    },
    BuiltinDocumentationFaq {
        question: "Is tand(X) equivalent to tan(X*pi/180)?",
        answer: "They are mathematically equivalent away from the poles, but the direct expression loses exact canonical results because `pi/180` is approximate.",
    },
    BuiltinDocumentationFaq {
        question: "Does tand support arrays?",
        answer: "Yes. It operates elementwise and preserves the input shape.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sind",
        target: BuiltinDocumentationLinkTarget::Builtin("sind"),
    },
    BuiltinDocumentationLink {
        label: "cosd",
        target: BuiltinDocumentationLinkTarget::Builtin("cosd"),
    },
    BuiltinDocumentationLink {
        label: "tan",
        target: BuiltinDocumentationLinkTarget::Builtin("tan"),
    },
    BuiltinDocumentationLink {
        label: "deg2rad",
        target: BuiltinDocumentationLinkTarget::Builtin("deg2rad"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tand.rs",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Degree tangent runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tand.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Canonical, wide-integer, complex, provider, and error behavior",
        location: "crates/runmat-runtime/src/builtins/math/trigonometry/tand.rs::tests",
    }],
    notes: &[],
};
pub(crate) const TAND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tand"),
    slug: Some("tand"),
    summary: "Compute elementwise tangent for degree-valued angles with exact canonical values and poles.",
    description: "`tand` evaluates degree-scaled tangent while recognizing canonical real phases before general floating evaluation.",
    keywords: &[
        "tand",
        "tangent",
        "degrees",
        "angle",
        "trigonometry",
        "exact",
        "poles",
        "elementwise",
        "gpu",
    ],
    related: &["sind", "cosd", "tan", "deg2rad"],
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
