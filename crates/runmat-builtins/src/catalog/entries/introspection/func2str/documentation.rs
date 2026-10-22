use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("func2str"),
    slug: Some("func2str"),
    summary: "Return the stored function-name text of a function handle.",
    description: "`name = func2str(fh)` returns a RunMat string scalar containing the name stored by a named, external, method, bound, or closure function handle. It reads handle metadata without calling the function or resolving an executable target.",
    keywords: &["func2str", "function handle", "function name", "introspection"],
    related: &["str2func", "functions"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported behavior",
            paragraphs: &[
                "Exactly one function handle is required. Numeric values, text, cells, objects, and provider-resident numeric values are not converted to handles. Non-handle inputs reject before provider access.",
                "Named, qualified, method, and bound handles return their stored name. A closure returns its stored internal function name; captured values are not included in the text.",
                "The operation does not invoke a callback, load a function, or grant the returned text any execution authority. Converting the text back with str2func does not preserve a bound handle's original source authority or a closure's captured workspace.",
                "Zero or one output is supported. Asking for more than one output rejects before inspecting handle metadata.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "MATLAB compatibility limits",
            paragraphs: &[
                "MATLAB returns a character vector. RunMat currently returns a string scalar, so callers that need a character vector must convert the result explicitly.",
                "MATLAB represents anonymous functions as expression text. RunMat currently returns a closure's stored internal function name and does not reconstruct an anonymous expression. Text round-tripping is therefore not a way to serialize closures.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &[
                "Handle text is host metadata. func2str has no numeric kernel, gather, device output, or numeric precision conversion. Existing text ownership is retained without copying captured workspace values.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "named-handle",
            title: "Inspect a named handle",
            program: "f = @sin;\nname = func2str(f);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(name) && isscalar(name) && strcmp(name, 'sin'));\nassert(abs(f(pi/2)-1)<1e-12);" },
        },
        BuiltinExample {
            id: "unresolved-name",
            title: "Read a name without resolving or invoking it",
            program: "f = str2func('func2str_unregistered_example_target');\nname = func2str(f);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(name) && strcmp(name, 'func2str_unregistered_example_target'));" },
        },
        BuiltinExample {
            id: "closure-name",
            title: "Inspect a closure without executing its body",
            program: "f = @(x) error('func2str must not invoke this callback');\nname = func2str(f);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(name) && isscalar(name) && strlength(name)>0);" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB func2str reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/func2str.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Function-handle text implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/function_handle_text.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Handle-text and output-bound tests",
            location: "crates/runmat-runtime/src/builtins/introspection/function_handle_text.rs::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
