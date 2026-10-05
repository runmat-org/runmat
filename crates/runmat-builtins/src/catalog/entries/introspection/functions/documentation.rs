use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("functions"), slug: Some("functions"),
    summary: "Inspect a function handle's current RunMat metadata.",
    description: "`functions(fh)` returns a scalar struct with `function`, `type`, and `file` fields for a function handle or closure. This is diagnostic metadata, not a stable serialization format.",
    keywords: &["functions", "function handle", "closure", "introspection"],
    related: &["localfunctions", "inputname"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Named, external, method, and bound handles report their name and `simple` type. Anonymous closures report `anonymous`; other closures report `scopedfunction`. The current implementation reports an empty `file` field because source-file provenance is not retained on these value variants.",
            "All three fields are RunMat string scalars. MATLAB's metadata can include additional fields and file paths; this implementation does not claim that broader representation.",
            "Non-handle inputs raise `RunMat:FunctionsHandleUnsupported` before resident-provider access.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "This portable native/browser metadata query does not invoke the handle or its captured values. It does not gather a resident payload and has no whole-call transfer schedule of its own.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "named-handle", title: "Inspect a named handle",
        program: "fh = @sin; info = functions(fh);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isstruct(info)); assert(strcmp(info.function,'sin')); assert(strcmp(info.type,'simple')); assert(strcmp(info.file,''));",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Can I rely on the metadata for program logic?",
        answer: "No. The structure is intended for inspection; source provenance and additional MATLAB fields are not currently represented.",
    }],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB functions reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/functions.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime handle classification",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/function_metadata.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Handle metadata and rejection",
            location: "builtins::introspection::function_metadata::tests",
        }, BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Compiled named, local, and anonymous handles",
            location: "crates/runmat-vm/tests/functions.rs::functions_builtin_reports_function_handle_metadata",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
