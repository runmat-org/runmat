use crate::*;

pub(super) const ISDEPLOYED_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"isdeployed"####),
    slug: Some(r####"isdeployed"####),
    summary: r####"Return false because RunMat is not MATLAB Compiler deployed code."####,
    description: r####"`isdeployed` returns a logical scalar indicating whether code is running as a MATLAB Compiler deployed application. RunMat always returns false."####,
    keywords: &[
        r####"isdeployed"####,
        r####"deployment"####,
        r####"runtime"####,
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and placement",
            paragraphs: &[
                r####"The builtin accepts no input arguments."####,
                r####"The result is always false in RunMat."####,
                r####"GPU support: Host metadata query."####,
                r####"The result is a host logical scalar."####,
                r####"No GPU kernels are launched."####,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Legacy examples and displayed results",
            paragraphs: &[r####"Checking deployment status
isdeployed
Expected display: ans = logical(0)"####],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "typed-basic",
            title: "check deployment status",
            program: "tf = isdeployed",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(~tf);",
            },
        },
        BuiltinExample {
            id: "arity-rejection",
            title: "Reject deployment arguments",
            program: "isdeployed(1)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:runtimeCompat:InvalidInput",
            },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: r####"Can RunMat run MATLAB Compiler deployed code?"####,
        answer: r####"No. This predicate exists so compatibility branches can execute deterministically."####,
    }],
    links: &[BuiltinDocumentationLink {
        label: r####"version"####,
        target: BuiltinDocumentationLinkTarget::Documentation(r####"./version"####),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: r####"`crates/runmat-runtime/src/builtins/introspection/runtime_compat.rs`"####,
            target: BuiltinDocumentationLinkTarget::Source(
                r####"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/runtime_compat.rs"####,
            ),
        },
        BuiltinDocumentationLink { label: "Catalog contract", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/runtime_compat/contract.rs") },
        BuiltinDocumentationLink { label: "Catalog inference", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/runtime_compat/inference.rs") }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Legacy test reference",
            location: r####"builtins::introspection::runtime_compat::tests"####,
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
