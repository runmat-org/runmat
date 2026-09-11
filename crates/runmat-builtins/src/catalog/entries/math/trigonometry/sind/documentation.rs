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
            "`sind(X)` evaluates sine with `X` measured in degrees. It returns exact zero at multiples of 180, exact positive or negative one at odd multiples of 90, and exact positive or negative one-half at angles congruent to 30 or 150 degrees with the corresponding sign. Other finite values use the degree-scaled analytic function; non-finite real input returns `NaN`.",
            "The operation is elementwise and preserves scalar, vector, matrix, empty, and N-D shape. Real and complex `single` input returns `single`; real and complex `double` input returns `double`. Complex input uses the analytic continuation after scaling by `pi/180`; canonical snapping applies only to real input.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat mode accepts all eight real fixed-width integer classes when each value is exactly representable at the binary64 trigonometric boundary. The result is double. Values that would lose integer information are rejected rather than rounded before degree reduction.",
            "Logical input is a separately gated RunMat extension and returns double. Character arrays, strings, sparse arrays, and typed complex integers are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &["Provider-resident input gathers through its owning provider and returns a host result so the canonical-angle implementation remains authoritative. Fusion is disabled because lowering to `sin(X*pi/180)` would lose exact canonical results."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "canonical-angle",
        title: "Evaluate a canonical angle",
        program: "y = sind(30)",
        display_output: Some("y = 0.5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 0.5);",
        },
    },
    BuiltinExample {
        id: "exact-multiples",
        title: "Return exact zeros at full half-turns",
        program: "y = sind([0 180 360])",
        display_output: Some("y = [0 0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [0 0 0]));",
        },
    },
    BuiltinExample {
        id: "common-angles",
        title: "Evaluate common degree angles",
        program: "angles = [0 30 45 60 90];\ny = sind(angles)",
        display_output: Some("y = [0 0.5 0.7071 0.8660 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "expected = [0 0.5 sqrt(0.5) sqrt(3)/2 1];\nassert(max(abs(y - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "single-class",
        title: "Preserve single precision",
        program: "y = sind(single([30 90]))",
        display_output: Some("y is a single array [0.5 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, \"single\"));\nassert(isequal(y, single([0.5 1])));",
        },
    },
    BuiltinExample {
        id: "provider-gather",
        title: "Gather provider input for canonical evaluation",
        program: "G = gpuArray([0 30 90]);\ny = sind(G)",
        display_output: Some("y = [0 0.5 1] on the host"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~isa(y, \"gpuArray\"));\nassert(isequal(y, [0 0.5 1]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does sind(180) return exactly zero?",
        answer: "RunMat reduces real angles modulo 360 and handles canonical cases before the floating sine calculation, avoiding noise from an approximate value of `pi`.",
    },
    BuiltinDocumentationFaq {
        question: "Is sind(X) equivalent to sin(X*pi/180)?",
        answer: "They are mathematically equivalent, but the direct expression can contain rounding noise at canonical angles. `sind` returns the defined exact values there.",
    },
    BuiltinDocumentationFaq {
        question: "Does sind support arrays?",
        answer: "Yes. It operates elementwise and preserves the input shape.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "cosd",
        target: BuiltinDocumentationLinkTarget::Builtin("cosd"),
    },
    BuiltinDocumentationLink {
        label: "tand",
        target: BuiltinDocumentationLinkTarget::Builtin("tand"),
    },
    BuiltinDocumentationLink {
        label: "sin",
        target: BuiltinDocumentationLinkTarget::Builtin("sin"),
    },
    BuiltinDocumentationLink {
        label: "deg2rad",
        target: BuiltinDocumentationLinkTarget::Builtin("deg2rad"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sind.rs"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Degree sine runtime",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sind.rs"),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Canonical, typed, complex, and error behavior",
        location: "crates/runmat-runtime/src/builtins/math/trigonometry/sind.rs::tests",
    }],
    notes: &[],
};
pub(crate) const SIND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sind"),
    slug: Some("sind"),
    summary: "Compute elementwise sine for degree-valued angles with exact canonical results.",
    description:
        "`sind` evaluates degree-scaled sine and handles canonical real angles before the general floating calculation.",
    keywords: &[
        "sind",
        "sine",
        "degrees",
        "angle",
        "trigonometry",
        "exact",
        "elementwise",
        "gpu",
    ],
    related: &["cosd", "tand", "sin", "deg2rad"],
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
