use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/outliers.rs",
    ),
};

pub(super) const ISOUTLIER_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isoutlier"), slug: Some("isoutlier"),
    summary: "Detect outliers in numeric arrays.",
    description: "`isoutlier(A)` returns a logical mask for observations outside method-specific lower and upper thresholds. Optional outputs expose the thresholds and center used for detection.",
    keywords: &["isoutlier", "outlier", "median", "quartiles", "mean", "statistics"],
    related: &["filloutliers", "fillmissing"],
    sections: &[
        BuiltinDocumentationSection { heading: "Detection methods", paragraphs: &[
            "The default `\"median\"` method uses three scaled median absolute deviations from the median. `\"quartiles\"` uses 1.5 interquartile ranges by default. `\"mean\"`, `\"percentiles\"`, `\"movmedian\"`, and `\"movmean\"` are also supported for numeric arrays.",
            "The `\"percentiles\"` method accepts a two-element percentile vector and does not accept `ThresholdFactor`. Other supported methods accept `ThresholdFactor`. An optional positive dimension chooses the operating axis.",
            "`[TF,L,U,C] = isoutlier(...)` returns the logical mask plus lower threshold, upper threshold, and center.",
        ] },
        BuiltinDocumentationSection { heading: "Classes and unsupported forms", paragraphs: &[
            "Single and double input arrays are supported. Typed-integer input data, dimensions, moving windows, and threshold controls are independently gated RunMat extensions. Integer values entering floating detection must be exactly representable as double.",
            "Table, timetable, `SamplePoints`, `DataVariables`, and `OutlierLocations` forms are not supported by this numeric-array implementation.",
        ] },
        BuiltinDocumentationSection { heading: "Resident arrays", paragraphs: &[
            "Floating resident input uses the owning provider's host-materialization path. Supported explicit `gpuArray` forms restore the logical mask and requested numeric thresholds through that same provider. This is not a resident outlier-detection kernel.",
            "Automatic internal residency can gather transparently. Explicit GPU `movmedian`, `SamplePoints`, and `DataVariables` restrictions are checked before gathering. A provider that cannot preserve explicit output residency returns an error rather than silently returning host results.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "vector", title: "Detect a vector outlier",
            program: "TF = isoutlier([1;2;100;4;5]); disp(TF);",
            display_output: Some("[false; false; true; false; false]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(TF, logical([0;0;1;0;0])));" },
        },
        BuiltinExample { id: "quartiles", title: "Return quartile thresholds",
            program: "A = [1;2;100;4;5]; [TF,L,U,C] = isoutlier(A,\"quartiles\"); disp(TF);",
            display_output: Some("The third observation is marked as an outlier."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(TF, logical([0;0;1;0;0]))); assert(L < U); assert(C >= L && C <= U);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which detection method is the default?", answer: "`median`, with a threshold of three scaled median absolute deviations." },
        BuiltinDocumentationFaq { question: "What does the first output contain?", answer: "A logical mask with the same shape as the input array." },
        BuiltinDocumentationFaq { question: "Does detection run on the GPU?", answer: "The current implementation materializes resident input on the host, then restores supported outputs through the input's provider." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "filloutliers", target: BuiltinDocumentationLinkTarget::Builtin("filloutliers") },
        BuiltinDocumentationLink { label: "fillmissing", target: BuiltinDocumentationLinkTarget::Builtin("fillmissing") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
            label: "Detection methods, integer exactness, thresholds, and resident restoration",
            location: "crates/runmat-runtime/src/builtins/stats/summary/outliers/tests.rs" }],
        notes: &["The runtime evaluates detection after owning-provider materialization. The supported explicit floating GPU route restores the mask and requested thresholds rather than running a detection kernel on device."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const FILLOUTLIERS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("filloutliers"), slug: Some("filloutliers"),
    summary: "Detect and replace outliers in numeric arrays.",
    description: "`filloutliers(A,fillmethod)` detects outlying observations and replaces them with a selected fill rule. Optional outputs report which entries were actually filled and the detection thresholds.",
    keywords: &["filloutliers", "outlier", "fill", "median", "quartiles", "mean", "statistics"],
    related: &["isoutlier", "fillmissing"],
    sections: &[
        BuiltinDocumentationSection { heading: "Detection and replacement", paragraphs: &[
            "Detection methods are `\"median\"` (default), `\"mean\"`, `\"quartiles\"`, `\"percentiles\"`, `\"movmedian\"`, and `\"movmean\"`. Fill methods are `\"center\"`, `\"clip\"`, `\"previous\"`, `\"next\"`, `\"nearest\"`, `\"linear\"`, or a numeric scalar constant.",
            "`[B,TF,L,U,C]` returns the filled array, actually filled-entry logical mask, lower and upper thresholds, and detection center. A logical `OutlierLocations` mask must exactly match the input shape; a numeric mask is a separately gated RunMat extension. `OutlierLocations` cannot be combined with an explicitly selected detection method.",
        ] },
        BuiltinDocumentationSection { heading: "Classes and boundaries", paragraphs: &[
            "Documented single and double input preserve their class in `B`. Typed-integer input is a RunMat-only form: it crosses into double statistics and output only if the exact input span is at most 2^53. Wider spans are rejected so distinct observations do not collapse at the floating boundary. `TF` remains logical; `L`, `U`, and `C` are double.",
            "Typed-integer constant fill scalars, numeric `OutlierLocations` masks, and direct provider-resident inputs or arguments are independent RunMat extensions classified before gather. The current fill calculation returns host output; unlike `isoutlier`, it does not restore requested outputs to the input provider.",
            "Spline-like interpolation, Grubbs/GESD detection, vector or duration windows, `SamplePoints`, `MaxNumOutliers`, and table/timetable options are unsupported instead of approximated.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "linear", title: "Linearly fill an outlier",
            program: "B = filloutliers([1;2;100;4;5],\"linear\"); disp(B);",
            display_output: Some("[1;2;3;4;5]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1;2;3;4;5]));" },
        },
        BuiltinExample { id: "constant", title: "Fill an outlier with a constant",
            program: "B = filloutliers([1;2;100;4;5],0); disp(B);",
            display_output: Some("[1;2;0;4;5]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1;2;0;4;5]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does TF mean?", answer: "It marks entries actually replaced in `B`, rather than every candidate outlier when no fill was possible." },
        BuiltinDocumentationFaq { question: "Are wide integers accepted?", answer: "Only if the exact input span is at most 2^53; admitted integer data produces double values in the filled result." },
        BuiltinDocumentationFaq { question: "Does filloutliers return a resident array?", answer: "The current implementation materializes resident input and returns a host result." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "isoutlier", target: BuiltinDocumentationLinkTarget::Builtin("isoutlier") },
        BuiltinDocumentationLink { label: "fillmissing", target: BuiltinDocumentationLinkTarget::Builtin("fillmissing") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
            label: "Replacement, output mask, integer spans, and unsupported forms",
            location: "crates/runmat-runtime/src/builtins/stats/summary/outliers/tests.rs" }],
        notes: &["The current fill calculation uses host materialization and does not promise provider-resident execution or output restoration."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
