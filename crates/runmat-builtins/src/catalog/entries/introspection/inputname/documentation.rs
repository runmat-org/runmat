use crate::*;

const NAMED_INPUT_FILE: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "named-caller",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "probe_input_name.m",
            content: BuiltinFixtureContent::Utf8(
                "function name = probe_input_name(value)\nname = inputname(1);\nend\n",
            ),
        }],
    });

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("inputname"), slug: Some("inputname"),
    summary: "Return the source variable name of a function input.",
    description: "`inputname(argNumber)` inspects the caller expression for a one-based function input index. A simple variable yields a character row; an expression, literal, or indexed value yields an empty 0x0 character array.",
    keywords: &["inputname", "function input", "caller", "introspection"],
    related: &["functions", "localfunctions"],
    sections: &[
        BuiltinDocumentationSection { heading: "Caller semantics", paragraphs: &[
            "The index must be a finite, real, positive integer-valued scalar in one of the documented numeric classes. Logical and resident indices are rejected before provider access.",
            "The runtime reads the function-input callsite's source spans. If a cell or dot indexing expression introduces a comma-separated-list input, that input and subsequent inputs have empty names. A call without function-input callsite metadata raises `RunMat:InputnameArgumentExceedsInputs`.",
            "The implementation does not yet reproduce MATLAB's stack walk through built-in, MEX, or overloaded indexing frames; this depends on a richer shared caller-metadata contract.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The query uses Core's current callsite/source catalog and does not invoke an operand callback or gather a resident input. Native and browser runtimes use the same source-span path.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "named-input", title: "Inspect a named function input",
        program: "input_value = 7; name = probe_input_name(input_value);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: NAMED_INPUT_FILE,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(name)); assert(strcmp(name,'input_value'));",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "What is returned for a literal or expression?",
        answer: "An empty 0x0 character array is returned when the caller argument has no simple variable name.",
    }],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB inputname reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/inputname.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Caller-span runtime implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/inputname.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Caller span and exact integer-index tests",
            location: "builtins::introspection::inputname::tests",
        }, BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Compiled direct and feval caller names",
            location: "crates/runmat-vm/tests/functions.rs::inputname_reports_direct_caller_argument_names",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
