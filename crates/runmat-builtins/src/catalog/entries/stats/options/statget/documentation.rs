use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("statget"), slug: Some("statget"),
    summary: "Read values from statistics options structures.",
    description: "`statget(options,field)` reads a canonical field by case-insensitive name or unique leading prefix. `defaultData` is returned only when a recognized field is empty. The default is an opaque value and retains its native class, shape, and payload.",
    keywords: &["statget", "statset", "statistics options"], related: &["statset"],
    sections: &[
        BuiltinDocumentationSection { heading: "Lookup and defaults", paragraphs: &[
            "The first argument must be a structure. An unknown or ambiguous field name returns an empty numeric array for loose option probing. A recognized but empty field returns its empty value unless defaultData is supplied.",
            "Only a selected empty field causes defaultData to be materialized. Native integers, nested values, and callback handles are returned without numeric conversion. A stored callback is not invoked by statget.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Lookup is host metadata work, not a GPU or fused operation. The field name is validated before reading the selected value; unselected fields are not materialized. A selected resident value or automatically resident default gathers through its owner. A stored foreign or worker-local handle may restrict subsequent transport or use.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "explicit", title: "Read an explicit option", program: "opts = statset('TolX', 1e-8);\ntol = statget(opts, 'TolX')", display_output: Some("tol = 1e-8"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tol,1e-8));" } },
        BuiltinExample { id: "default", title: "Use a default for an empty option", program: "opts = statset('MaxIter', 100);\ntol = statget(opts, 'TolFun', 1e-6)", display_output: Some("tol = 1e-6"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tol,1e-6));" } },
        BuiltinExample { id: "prefix", title: "Use a unique field prefix", program: "opts = statset('MaxIter', 100);\niter = statget(opts, 'MaxI')", display_output: Some("iter = 100"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(iter,100));" } },
    ],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "statset", target: BuiltinDocumentationLinkTarget::Builtin("statset") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Statistics options runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/options.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Statistics options lookup and exact defaults", location: "crates/runmat-runtime/src/builtins/stats/options/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled statistics options calls", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
