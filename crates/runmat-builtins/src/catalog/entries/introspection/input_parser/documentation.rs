use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("inputParser"),
    slug: Some("inputParser"),
    summary: "Parse MATLAB-style optional name-value arguments.",
    description: "`inputParser` creates a handle object for parsing common MATLAB name-value argument patterns. RunMat supports the minimal workflow used by imported helper functions: create a parser, register parameters with defaults, parse `varargin{:}`, and read values from `p.Results`.",
    keywords: &[
        "inputParser",
        "addParameter",
        "parse",
        "name-value",
        "varargin",
        "arguments",
    ],
    related: &[
        "inputParser",
        "addParameter",
        "parse",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`p = inputParser` returns a handle object with an initially empty `Results` struct.",
                "`addParameter(p, name, default)` registers a case-sensitive name and default value, mutating `p` in place.",
                "`parse(p, name, value, ...)` starts from the registered defaults and applies exact name matches.",
                "`p.Results.<name>` reads the parsed value for a registered parameter.",
                "Duplicate parameter names, odd name-value argument counts, non-text names, and unknown parsed names raise stable RunMat errors.",
                "This compatibility slice does not yet implement `addRequired`, `addOptional`, validators, `CaseSensitive`, `PartialMatching`, `KeepUnmatched`, or `StructExpand`.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "inputParser stores host-side metadata and parsed values.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Supported parser contract",
            paragraphs: &[
                "inputParser supports case-sensitive named parameters with defaults. addParameter registers a parameter and rejects duplicates. parse accepts complete name-value pairs, rejects unknown names, uses the last supplied value for duplicates, and publishes Results only after validation. The current implementation does not support validators, positional parameters, or parser configuration options.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Parse an optional scale parameter",
            program: "result = f('scale', 4);\nfunction y = f(varargin)\n  p = inputParser;\n  addParameter(p, 'scale', 2);\n  parse(p, varargin{:});\n  y = p.Results.scale;\nend",
            display_output: Some("ans =\n     4"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(result == 4);",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use a registered default when the name is omitted",
            program: "p = inputParser;\naddParameter(p, 'scale', 2);\nparse(p);\np.Results.scale",
            display_output: Some("ans =\n     2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(p.Results.scale == 2);",
            },
        },
        BuiltinExample {
            id: "named-default",
            title: "Parse a named parameter",
            program: "p = inputParser(); addParameter(p, 'Count', 2); parse(p, 'Count', 5); result = p.Results.Count;",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(result == 5);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Are parameter names case-sensitive?",
            answer: "Yes. This initial RunMat implementation uses exact case-sensitive name matching.",
        },
        BuiltinDocumentationFaq {
            question: "Does RunMat support validators or positional arguments?",
            answer: "Not yet. The current support is intentionally limited to optional name-value parameters registered with `addParameter`.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "inputParser",
            target: BuiltinDocumentationLinkTarget::Builtin("inputParser")
        },
        BuiltinDocumentationLink {
            label: "addParameter",
            target: BuiltinDocumentationLinkTarget::Builtin("addParameter")
        },
        BuiltinDocumentationLink {
            label: "parse",
            target: BuiltinDocumentationLinkTarget::Builtin("parse")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::introspection::input_parser::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "runmat-vm inputParser tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
