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
            "`cosd(X)` evaluates cosine with `X` measured in degrees. It returns exact zero at odd multiples of 90, exact positive or negative one at multiples of 180, and exact positive or negative one-half at angles congruent to 60 or 120 degrees with the corresponding sign. Other finite values use the degree-scaled analytic function; non-finite real input returns `NaN`.",
            "The operation is elementwise and preserves scalar, vector, matrix, empty, and N-D shape. Real and complex `single` input returns `single`; real and complex `double` input returns `double`. Complex input uses the analytic continuation after scaling by `pi/180`; canonical snapping applies only to real input.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat mode accepts all eight real fixed-width integer classes when each value is exactly representable at the binary64 trigonometric boundary. The result is double; aligned values above `flintmax` remain accepted when conversion is exact, while values that would lose information are rejected.",
            "Logical and character arrays are separately gated RunMat extensions and return double. Character input is evaluated from exact Unicode scalar values. Strings, sparse arrays, typed complex integers, tables, timetables, and tall containers are rejected. Execution-owned distributed arrays use the catalog's unary mapping policy.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "Provider-resident input gathers through its owner for canonical host evaluation, then the result is uploaded to that same owner. Fusion is disabled because lowering to `cos(X*pi/180)` would lose exact canonical results.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "canonical-angle",
        title: "Evaluate a canonical angle",
        program: "y = cosd(60)",
        display_output: Some("y = 0.5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 0.5);",
        },
    },
    BuiltinExample {
        id: "exact-quarter-turns",
        title: "Return exact zeros at odd quarter-turns",
        program: "y = cosd([90 270])",
        display_output: Some("y = [0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [0 0]));",
        },
    },
    BuiltinExample {
        id: "common-angles",
        title: "Evaluate common degree angles",
        program: "angles = [0 30 45 60 90];\ny = cosd(angles)",
        display_output: Some("y = [1 0.8660 0.7071 0.5 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1 sqrt(3)/2 sqrt(0.5) 0.5 0];\nassert(max(abs(y - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-extension",
        title: "Evaluate character code points in RunMat mode",
        program: "y = cosd('AZ')",
        display_output: Some("y contains cosd(65) and cosd(90)"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, \"double\"));\nassert(y(2) == 0);\nassert(abs(y(1) - cosd(65)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "provider-restoration",
        title: "Restore canonical results to the input provider",
        program: "G = gpuArray([0 60 90]);\ny = cosd(G);\nhostY = gather(y)",
        display_output: Some("hostY = [1 0.5 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, \"gpuArray\"));\nassert(isequal(hostY, [1 0.5 0]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does cosd(90) return exactly zero?",
        answer: "RunMat reduces real angles modulo 360 and handles canonical cases before the floating cosine calculation, avoiding noise from an approximate value of `pi`.",
    },
    BuiltinDocumentationFaq {
        question: "Is cosd(X) equivalent to cos(X*pi/180)?",
        answer: "They are mathematically equivalent, but the direct expression can contain rounding noise at canonical angles. `cosd` returns the defined exact values there.",
    },
    BuiltinDocumentationFaq {
        question: "Does cosd support arrays?",
        answer: "Yes. It operates elementwise and preserves the input shape.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "MATLAB cosd documentation",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.cosd.html",
        ),
    },
    BuiltinDocumentationLink {
        label: "sind",
        target: BuiltinDocumentationLinkTarget::Builtin("sind"),
    },
    BuiltinDocumentationLink {
        label: "tand",
        target: BuiltinDocumentationLinkTarget::Builtin("tand"),
    },
    BuiltinDocumentationLink {
        label: "cos",
        target: BuiltinDocumentationLinkTarget::Builtin("cos"),
    },
    BuiltinDocumentationLink {
        label: "deg2rad",
        target: BuiltinDocumentationLinkTarget::Builtin("deg2rad"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cosd.rs"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Degree cosine runtime",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cosd.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Canonical, typed, complex, and error behavior",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/cosd.rs::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Owner-preserving host fallback",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/cosd.rs::tests::gpu_fallback_restores_output_to_owner",
        },
    ],
    notes: &[],
};
pub(crate) const COSD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cosd"),
    slug: Some("cosd"),
    summary: "Compute elementwise cosine for degree-valued angles with exact canonical results.",
    description: "`cosd` evaluates degree-scaled cosine and handles canonical real angles before the general floating calculation.",
    keywords: &["cosd", "cosine", "degrees", "angle", "trigonometry", "exact", "elementwise", "gpu"],
    related: &["sind", "tand", "cos", "deg2rad"],
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
