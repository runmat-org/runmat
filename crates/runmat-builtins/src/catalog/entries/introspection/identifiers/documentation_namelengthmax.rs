use crate::*;

pub(super) const NAMELENGTHMAX_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"namelengthmax"####),
    slug: Some(r####"namelengthmax"####),
    summary: r####"Return the maximum MATLAB identifier length."####,
    description: r####"`namelengthmax` returns the maximum number of characters allowed in MATLAB identifiers. RunMat reports `2048`, matching current MATLAB behavior."####,
    keywords: &[
        r####"namelengthmax"####,
        r####"identifier"####,
        r####"variable name"####,
        r####"field name"####,
        r####"MATLAB compatibility"####,
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and placement",
            paragraphs: &[
                r####"Accepts no input arguments and returns a double scalar."####,
                r####"The shared identifier validator uses the same value for `isvarname`, `genvarname`, struct field-name validation, and argument-validation helpers."####,
                r####"Identifiers longer than `namelengthmax` are rejected by `isvarname` and repaired or truncated by compatibility helpers that generate names."####,
                r####"GPU support: Host language constant; no GPU kernels or provider dispatch are involved."####,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Legacy examples and displayed results",
            paragraphs: &[
                r####"Read the identifier length limit
l = namelengthmax
Expected display: l = 2048"####,
                r####"Use the limit when validating generated names
name = repmat('a', 1, namelengthmax);
tf = isvarname(name)
Expected display: tf = logical(1)"####,
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "typed-basic",
            title: "query identifier bound",
            program: "n = namelengthmax",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n > 0);",
            },
        },
        BuiltinExample {
            id: "arity-rejection",
            title: "Reject input arguments",
            program: "namelengthmax(1)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:namelengthmax:InvalidArgument",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: r####"Why is the value 2048 rather than 63?"####,
            answer: r####"Older MATLAB releases used 63. Current MATLAB releases report 2048, and RunMat follows the current compatibility surface."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does this run on the GPU?"####,
            answer: r####"No. This is a language metadata constant and always returns a host scalar."####,
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: r####"isvarname"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./isvarname"####),
        },
        BuiltinDocumentationLink {
            label: r####"iskeyword"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./iskeyword"####),
        },
        BuiltinDocumentationLink {
            label: r####"genvarname"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./genvarname"####),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: r####"`crates/runmat-runtime/src/builtins/introspection/identifiers.rs`"####,
            target: BuiltinDocumentationLinkTarget::Source(
                r####"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/identifiers.rs"####,
            ),
        },
        BuiltinDocumentationLink { label: "Catalog contract", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/identifiers/contract.rs") },
        BuiltinDocumentationLink { label: "Catalog inference", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/identifiers/inference.rs") }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Legacy test reference",
            location: r####"builtins::introspection::identifiers::tests"####,
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
