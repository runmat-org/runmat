use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Conversion",
        paragraphs: &[
            "`deg2rad(D)` converts each element from degrees to radians by multiplying by `pi/180`. Scalar, vector, matrix, empty, and N-D shapes are preserved.",
            "Real and complex `single` input returns `single`; real and complex `double` input returns `double`. Complex values are scaled component by component, as with ordinary multiplication by the real conversion factor.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat mode also accepts real fixed-width integer and logical input. Integers must be exactly representable as binary64 before conversion, and both extensions return double. MATLAB compatibility mode rejects these extension-only forms.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Floating provider-resident input can participate in elementwise fusion. When host evaluation is required, RunMat gathers through the exact owning provider and restores a validated result with the input's placement intent. It rejects a conversion when that contract cannot be preserved.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "right-angle",
        title: "Convert a right angle",
        program: "r = deg2rad(90)",
        display_output: Some("r = 1.5708"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(r - pi/2) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Convert a vector while preserving its shape",
        program: "d = [0 30 45 60 90];\nr = deg2rad(d)",
        display_output: Some("r = [0 0.5236 0.7854 1.0472 1.5708]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(r), size(d)));\nassert(max(abs(r - d*pi/180)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "single",
        title: "Preserve single precision",
        program: "r = deg2rad(single([0 90 180]))",
        display_output: Some("r is a single array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isa(r, \"single\"));\nassert(max(abs(double(r) - [0 pi/2 pi])) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "integer-extension",
        title: "Convert fixed-width integer angles in RunMat mode",
        program: "r = deg2rad(int16([0 90 180]))",
        display_output: Some("r = [0 1.5708 3.1416]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(r, \"double\"));\nassert(max(abs(r - [0 pi/2 pi])) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is deg2rad(D) equivalent to D*pi/180?",
        answer: "Yes. `deg2rad` names the unit conversion directly and applies it elementwise.",
    },
    BuiltinDocumentationFaq {
        question: "Does deg2rad preserve the input shape?",
        answer: "Yes. The output has the same shape as the input.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "rad2deg",
        target: BuiltinDocumentationLinkTarget::Builtin("rad2deg"),
    },
    BuiltinDocumentationLink {
        label: "Compatible deg2rad reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/deg2rad.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Angle conversion runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/deg2rad.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Shape, precision, complex, extension, error, and provider behavior",
        location: "crates/runmat-runtime/src/builtins/math/trigonometry/angle_conversion/deg2rad.rs",
    }],
    notes: &[],
};

pub(super) const DEG2RAD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("deg2rad"),
    slug: Some("deg2rad"),
    summary: "Convert angles from degrees to radians elementwise.",
    description: "`deg2rad` scales real or complex floating-point angles by `pi/180` while preserving shape and floating-point precision.",
    keywords: &["deg2rad", "degrees", "radians", "angle", "conversion"],
    related: &["rad2deg"],
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
