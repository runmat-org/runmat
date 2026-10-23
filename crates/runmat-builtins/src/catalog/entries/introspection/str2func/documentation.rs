use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("str2func"),
    slug: Some("str2func"),
    summary: "Construct a function handle from scalar function-name text.",
    description: "`str2func(name)` creates a function handle from a character row vector or scalar string. The text can name a builtin, user function, or qualified function; RunMat resolves known semantic functions and retains a late-bound name when resolution must happen later.",
    keywords: &["str2func",
        "function handle",
        "introspection",
        "text"],
    related: &["func2str",
        "functions"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["The input must be scalar row text. Nonscalar string arrays and multirow character arrays reject with a shape error.",
                "A leading `@` is accepted and removed before the name is stored, and surrounding whitespace is ignored.",
                "Empty names reject instead of creating an unusable handle.",
                "Integer and other numeric inputs are not converted to text. Provider-resident numeric values reject before RunMat attempts a gather.",
                "RunMat currently constructs named handles. Parsing the full text representation of an anonymous function remains an implementation gap."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Function-handle construction consumes host text. Numeric and provider-resident values reject before provider access."]
        },
        BuiltinDocumentationSection {
            heading: "Fusion boundary",
            paragraphs: &["Handle construction is an introspection operation and is not eligible for numeric fusion."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Create and call a named function handle",
            program: "f = str2func('sin');\ny = f(pi/2)",
            display_output: Some("y = 1"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(abs(y-1)<1e-12 && isa(f,'function_handle'));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `str2func` convert a numeric value into a function name?",
            answer: "No. It accepts scalar function-name text only. Convert data to text explicitly only when that is genuinely the intended program behavior."
        },
        BuiltinDocumentationFaq {
            question: "When is the named function resolved?",
            answer: "RunMat binds a known semantic function immediately when possible. Otherwise it keeps the validated name in the handle so ordinary runtime function resolution can select the target when the handle is called."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "func2str",
            target: BuiltinDocumentationLinkTarget::Documentation("./func2str")
        },
        BuiltinDocumentationLink {
            label: "functions",
            target: BuiltinDocumentationLinkTarget::Documentation("./functions")
        }
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/introspection/function_handle_text.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/function_handle_text.rs")
            }
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Runtime semantic tests",
            location: "crates/runmat-runtime/src/builtins/introspection/function_handle_text.rs::tests",
        }],
        notes: &[]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial)
}
;
