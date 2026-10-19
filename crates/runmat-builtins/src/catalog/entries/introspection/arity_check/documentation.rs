use crate::*;

const INPUT_FIXTURE: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "input-arity-function",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "arity_input_probe.m",
            content: BuiltinFixtureContent::Utf8(
                "function y = arity_input_probe(x)\nnarginchk(1,1);\ny = x;\nend\n",
            ),
        }],
    });
const OUTPUT_FIXTURE: BuiltinExampleFixture =
    BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "output-arity-function",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: &[BuiltinFilesystemEntry::File {
            relative_path: "arity_output_probe.m",
            content: BuiltinFixtureContent::Utf8(
                "function y = arity_output_probe(x)\nnargoutchk(1,1);\ny = x;\nend\n",
            ),
        }],
    });

pub(super) const NARGINCHK_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("narginchk"), slug: Some("narginchk"),
    summary: "Validate the number of inputs supplied to the current function.",
    description: "`narginchk(minArgs,maxArgs)` throws when the selected function received fewer than `minArgs` or more than `maxArgs` inputs. It produces no output; requesting a result reports `RunMat:TooManyOutputs`.",
    keywords: &["narginchk", "function arguments", "arity", "validation"],
    related: &["nargoutchk"],
    sections: &[
        BuiltinDocumentationSection { heading: "Bounds and context", paragraphs: &[
            "`minArgs` accepts a nonnegative scalar single, double, or any of the eight integer classes. `maxArgs` accepts the same classes and positive single or double infinity for an unbounded maximum. Integer storage is read exactly, without floating conversion; out-of-range counts reject rather than round or saturate.",
            "The current invocation's function input count is required. A top-level script or context-free embedding call has no function-call arity and reports `RunMat:NarginchkContextUnavailable` after argument-count admission.",
            "Interactive resident bounds are not documented and reject before provider access. The check is host-only and performs no provider operation or callback.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "one-input", title: "Require exactly one function input",
        program: "y = arity_input_probe(7);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: INPUT_FIXTURE,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(y == 7);" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "nargoutchk", target: BuiltinDocumentationLinkTarget::Builtin("nargoutchk") },
        BuiltinDocumentationLink { label: "Function input counts with nargin", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/docs/FUNCTIONS.md#optional-arguments-with-nargin") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime arity checker", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/arity_check.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact scalar integer and arity errors", location: "builtins::introspection::arity_check::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled function input arity", location: "crates/runmat-vm/tests/functions.rs::narginchk_nargoutchk_validate_function_arity" },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};

pub(super) const NARGOUTCHK_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nargoutchk"), slug: Some("nargoutchk"),
    summary: "Validate requested function outputs or return a legacy output-count message.",
    description: "`nargoutchk(minArgs,maxArgs)` throws when the current function's requested output count is outside the bounds. The discouraged three- and four-input forms validate an explicit count and return a message instead of throwing for a count violation.",
    keywords: &["nargoutchk", "function outputs", "arity", "validation"],
    related: &["narginchk"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and results", paragraphs: &[
            "The two-input form uses the selected function invocation's requested output count and produces no output; requesting a result reports `RunMat:TooManyOutputs`. A script or context-free embedding has no function output count and reports `RunMat:NargoutchkContextUnavailable` after its bounds are admitted.",
            "`msg = nargoutchk(minArgs,maxArgs,numArgs)` returns character message text on a violation and an empty 0-by-0 character array otherwise. With a fourth `'struct'` selector it returns a structure with `message` and `identifier` fields on a violation and an empty structure otherwise. These legacy forms return at most one result; a discarded call produces no result.",
            "`minArgs`, `maxArgs`, and legacy `numArgs` accept scalar single, double, or any of the eight integer classes. Positive single or double infinity is allowed only for `maxArgs`. Exact integer storage is compared without floating conversion; out-of-range counts reject rather than round or saturate.",
            "Interactive resident controls are not documented and reject before provider access. These forms are host-only and perform no provider operation or callback.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "one-output", title: "Require one requested function output",
            program: "y = arity_output_probe(7);", display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: OUTPUT_FIXTURE, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(y == 7);" },
        },
        BuiltinExample {
            id: "legacy-messages", title: "Inspect an explicit output-count violation",
            program: "msg = nargoutchk(1,1,0); detail = nargoutchk(1,1,0,'struct');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(msg,'Not enough output arguments.')); assert(strcmp(detail.identifier,'MATLAB:nargoutchk:notEnoughOutputs'));" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "narginchk", target: BuiltinDocumentationLinkTarget::Builtin("narginchk") },
        BuiltinDocumentationLink { label: "Function output counts with nargout", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/docs/FUNCTIONS.md#optional-arguments-with-nargin") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime arity checker", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/arity_check.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact integer and legacy text/struct result tests", location: "builtins::introspection::arity_check::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled function output arity", location: "crates/runmat-vm/tests/functions.rs::narginchk_nargoutchk_validate_function_arity" },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
