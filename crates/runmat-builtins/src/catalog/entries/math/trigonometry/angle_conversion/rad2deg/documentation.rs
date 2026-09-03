use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Conversion",
        paragraphs: &[
            "`rad2deg(R)` converts each element from radians to degrees by multiplying by `180/pi`. Scalar, vector, matrix, empty, and N-D shapes are preserved.",
            "Real and complex `single` input returns `single`; real and complex `double` input returns `double`. Complex values are scaled component by component.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &["RunMat mode also accepts real fixed-width integer and logical input. Integers must be exactly representable as binary64 before conversion, and both extensions return double. MATLAB compatibility mode rejects these extension-only forms."],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &["Floating provider-resident input can participate in elementwise fusion. When host evaluation is required, RunMat gathers through the exact owning provider and restores a validated result with the input's placement intent."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "straight-angle",
        title: "Convert a straight angle",
        program: "d = rad2deg(pi)",
        display_output: Some("d = 180"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(d - 180) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Convert a vector while preserving its shape",
        program: "r = [0 pi/6 pi/4 pi/3 pi/2];\nd = rad2deg(r)",
        display_output: Some("d = [0 30 45 60 90]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isequal(size(d), size(r)));\nassert(max(abs(d - [0 30 45 60 90])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "single",
        title: "Preserve single precision",
        program: "d = rad2deg(single([0 pi/2 pi]))",
        display_output: Some("d is a single array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(d, \"single\"));\nassert(max(abs(double(d) - [0 90 180])) < 1e-4);",
        },
    },
    BuiltinExample {
        id: "integer-extension",
        title: "Convert fixed-width integer radians in RunMat mode",
        program: "d = rad2deg(int16([0 1 2]))",
        display_output: Some("d is a double array"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(d, \"double\"));\nassert(max(abs(d - [0 1 2]*180/pi)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is rad2deg(R) equivalent to R*180/pi?",
        answer: "Yes. `rad2deg` names the unit conversion directly and applies it elementwise.",
    },
    BuiltinDocumentationFaq {
        question: "Does rad2deg preserve the input shape?",
        answer: "Yes. The output has the same shape as the input.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "deg2rad",
        target: BuiltinDocumentationLinkTarget::Builtin("deg2rad"),
    },
    BuiltinDocumentationLink {
        label: "Compatible rad2deg reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/rad2deg.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Angle conversion runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/rad2deg.rs") }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Shape, precision, complex, extension, error, and provider behavior",
        location: "crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/rad2deg.rs",
    }],
    notes: &[],
};

pub(super) const RAD2DEG_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rad2deg"),
    slug: Some("rad2deg"),
    summary: "Convert angles from radians to degrees elementwise.",
    description: "`rad2deg` scales real or complex floating-point angles by `180/pi` while preserving shape and floating-point precision.",
    keywords: &["rad2deg", "radians", "degrees", "angle", "conversion"],
    related: &["deg2rad"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("R2015b"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
