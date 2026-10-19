use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/normalize.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("normalize"), slug: Some("normalize"),
    summary: "Normalize data by centering and scaling along a dimension.",
    description: "`normalize(A)` returns z-scores along the first nonsingleton dimension. Select another method, operating dimension, or explicit center and scale with additional arguments.",
    keywords: &["normalize", "zscore", "scale", "range", "norm", "center", "statistics"],
    related: &["std", "mean"],
    sections: &[
        BuiltinDocumentationSection { heading: "Methods and outputs", paragraphs: &[
            "Default z-score normalization subtracts the mean and divides by the sample standard deviation, omitting NaN observations when calculating the parameters and retaining their positions in `N`. The optional positive `dim` selects the operating dimension.",
            "Methods include `\"zscore\"`, robust z-score, `\"norm\"` with optional positive order, `\"scale\"`, `\"range\"`, `\"center\"`, and `\"medianiqr\"`. Combined `\"center\", centertype, \"scale\", scaletype` forms also accept explicit numeric parameter arrays with compatible implicit expansion.",
            "`[N,C,S] = normalize(...)` returns the normalized array, center, and scale parameters for reuse. Parameter shapes depend on the method and any explicit parameters.",
        ] },
        BuiltinDocumentationSection { heading: "Classes and exactness", paragraphs: &[
            "Single input produces single `N`, `C`, and `S`; double input produces double outputs. Complex floating-point input is supported except for range normalization. Integer and logical input are separately gated RunMat extensions and produce double outputs.",
            "Admitted integer input data and numeric parameters must be exactly representable at the binary64 normalization boundary. Values that would lose integer precision are rejected rather than rounded.",
        ] },
        BuiltinDocumentationSection { heading: "Resident arrays", paragraphs: &[
            "Floating resident input currently uses class-preserving host calculation. Requested numeric outputs are restored through the input's exact provider owner when it can represent their class. An incompatible automatic owner may leave the correct-class result on the host; explicit `gpuArray` intent errors if residency cannot be preserved.",
            "Automatic resident controls may be gathered. Explicit resident control or parameter values are a gated RunMat extension and do not determine output residency. This is a host-materialization boundary, not a resident normalization kernel.",
            "Table and timetable `DataVariables` and `ReplaceValues` forms are not yet supported for array input.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "zscore", title: "Default z-score normalization",
            program: "N = normalize([1; 2; 3]); disp(N);",
            display_output: Some("[-1; 0; 1]"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(N), [3 1])); assert(all(abs(N - [-1; 0; 1]) < 1e-12));" },
        },
        BuiltinExample { id: "norm", title: "Normalize by Euclidean norm",
            program: "N = normalize([3; 4], \"norm\"); disp(N);",
            display_output: Some("[0.6; 0.8]"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(N), [2 1])); assert(all(abs(N - [0.6; 0.8]) < 1e-12));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does normalize do by default?", answer: "It subtracts the mean and divides by the sample standard deviation along the first nonsingleton dimension." },
        BuiltinDocumentationFaq { question: "Does single input stay single?", answer: "Yes. Single input produces single normalized data and center/scale parameters." },
        BuiltinDocumentationFaq { question: "Can normalize operate on gpuArray input?", answer: "The current implementation gathers for host calculation and restores supported outputs through the input's provider; it does not execute a resident normalization kernel." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "std", target: BuiltinDocumentationLinkTarget::Builtin("std") },
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
            label: "Normalization methods, exactness, precision, dimension, and provider restoration",
            location: "crates/runmat-runtime/src/builtins/stats/summary/normalize/tests.rs" }],
        notes: &["Normalization computes on the host and restores the output class and provider placement where supported by the input owner. It does not use a resident normalization kernel."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
