use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("head"),
    slug: Some("head"),
    summary: "Return the first rows of a table, timetable, or array.",
    description: "`head(A, n)` selects the first `n` rows, or the first eight rows when `n` is omitted. Numeric arrays retain their class and trailing dimensions.",
    keywords: &["head", "table", "timetable", "preview", "rows"],
    related: &["table", "timetable"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Table and timetable variable metadata and row order are retained.", "Real, complex, logical, and integer arrays retain every trailing dimension. Typed integer counts are decoded exactly and must be positive."] },
        BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &["A GPU-resident array is downloaded non-destructively and the selected value is restored to its owning provider with its class preserved.", "A GPU-resident row-count control is a separately gated RunMat extension; compatibility admission occurs before provider access. This operation does not run a GPU selection kernel."] },
    ],
    examples: &[
        BuiltinExample { id: "table-first-row", title: "Select the first table row", program: "T = table(uint64([10;20]), [1;2], 'VariableNames', {'Count','Score'}); H = head(T, 1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(istable(H)); assert(height(H)==1); assert(width(H)==2); assert(isequal(H.Count,uint64(10))); assert(isequal(H.Score,1));" } },
        BuiltinExample { id: "integer-pages", title: "Preserve integer array pages", program: "A = reshape(uint64(1:12), [3 2 2]); B = head(A, 2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(B,'uint64')); assert(isequal(size(B),[2 2 2])); assert(B(2,2,2)==uint64(11));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "table", target: BuiltinDocumentationLinkTarget::Builtin("table") }, BuiltinDocumentationLink { label: "timetable", target: BuiltinDocumentationLinkTarget::Builtin("timetable") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/selectors.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Head rows, integer precision, ND shape, and GPU restoration", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM integer and table selection integration", location: "crates/runmat-vm/tests/integer_multidimensional_fft_and_table_selection.rs" }],
        notes: &["The legacy display-only example is replaced by assertions over table class, shape, and contents.", "Legacy gpu_support and fusion fields are represented by typed placement and the GPU section. requires_feature was null; tested maps to test evidence."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
