use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fillmissing"),
    slug: Some("fillmissing"),
    summary: "Fill supported missing entries with a constant, neighbor, interpolation, or summary.",
    description: "`fillmissing(A, method, ...)` preserves the input shape and can also return a logical mask of entries actually filled.",
    keywords: &["fillmissing", "missing", "NaN", "string", "table", "constant", "previous", "next", "linear"],
    related: &["ismissing", "rmmissing", "standardizeMissing"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Methods and dimensions",
            paragraphs: &[
                r#"Numeric arrays support `"constant"`, `"previous"`, `"next"`, `"nearest"`, `"linear"`, `"mean"`, and `"median"`. Strings support constant and neighbor methods. Cells support constant replacement; tables are processed variable by variable."#,
                r#"The first non-singleton dimension is the default. Numeric and string arrays support any positive dimension, supplied directly or as `"dim", value`; each column-major line is filled independently, including N-D arrays. Dimensions beyond the array rank select singleton lines. The second output preserves the input shape and marks only entries actually filled; an entry left missing remains false."#,
                "Logical arrays are unchanged and return an all-false mask with the same shape. Complex and sparse array filling is unsupported. Spline, pchip, makima, mode, moving-window, KNN, custom-function, sample-point, known-location, maximum-gap, variable-selection, and append forms are not implemented.",
                "Mean and median fills use the non-missing values along each selected line. A line with no available values remains missing.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Integer and resident input",
            paragraphs: &[
                "A fixed-width integer array has no standard missing value. Its RunMat-only form returns the same exact integer array and an all-false mask. Integer table variables and nested cell values are a separate RunMat-only form; compatibility admission occurs before resident data is gathered.",
                "Resident inputs gather through their owning provider for filling. This implementation returns a host result and does not claim a GPU filling kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "linear-fill",
            title: "Fill by linear interpolation",
            program: "B = fillmissing([1 NaN 3], 'linear')",
            display_output: Some("B = [1 2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1 2 3]));" },
        },
        BuiltinExample {
            id: "constant-fill",
            title: "Fill with a constant",
            program: "B = fillmissing([1 NaN 3], 'constant', 0)",
            display_output: Some("B = [1 0 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1 0 3]));" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ismissing", target: BuiltinDocumentationLinkTarget::Builtin("ismissing") },
        BuiltinDocumentationLink { label: "rmmissing", target: BuiltinDocumentationLinkTarget::Builtin("rmmissing") },
        BuiltinDocumentationLink { label: "standardizeMissing", target: BuiltinDocumentationLinkTarget::Builtin("standardizeMissing") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Fill admission and output handling", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/entrypoints.rs") },
            BuiltinDocumentationLink { label: "Fill algorithms", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/missing/domain/filling/mod.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric, integer, aggregate, and admission semantics", location: "crates/runmat-runtime/src/builtins/missing/domain/tests/filling.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled missing-data calls", location: "crates/runmat-vm/tests/missing_values.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
