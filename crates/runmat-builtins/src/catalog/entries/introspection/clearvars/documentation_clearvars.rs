use crate::*;

pub(super) const CLEARVARS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"clearvars"####),
    slug: Some(r####"clearvars"####),
    summary: r####"Clear workspace variables with optional exclusions in MATLAB and RunMat."####,
    description: r####"`clearvars` removes variables from the active workspace, with optional `-except` exclusions. Named and exclusion forms follow MATLAB semantics for selectively clearing bindings."####,
    keywords: &[
        r####"clearvars"####,
        r####"clear variables"####,
        r####"workspace"####,
        r####"variables"####,
        r####"-except"####,
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and placement",
            paragraphs: &[
                r####"`clearvars` with no inputs clears all workspace variables created in the current session."####,
                r####"`clearvars x` removes the variable `x` while leaving other workspace bindings intact."####,
                r####"Multiple names can be supplied as separate inputs or command-form tokens, for example `clearvars x y z`."####,
                r####"`clearvars -except x y` removes all active workspace variables except `x` and `y`."####,
                r####"`clearvars a b c -except b` removes only selected variables `a` and `c`, preserves selected exclusion `b`, and leaves variables outside the selected clear set untouched."####,
                r####"Active `onCleanup` objects are invoked before removed workspace bindings are deleted; excluded cleanup objects remain active."####,
                r####"Function-call forms accept variable names as string scalars, row character vectors, or string arrays."####,
                r####"Clearing a name that does not exist is a no-op."####,
                r####"`clearvars` is a sink builtin and does not produce a meaningful output value."####,
                r####"GPU support: Runs on the host CPU because it mutates the session workspace rather than array contents."####,
                r####"`clearvars` performs host-side workspace mutation only. It does not launch GPU kernels, does not participate in fusion, and does not depend on acceleration providers."####,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Legacy examples and displayed results",
            paragraphs: &[
                r####"Clear the entire interactive workspace
x = 1;
y = magic(3);
clearvars
Expected display: % x and y are removed from the workspace"####,
                r####"Clear selected variables
a = 1;
b = 2;
c = 3;
clearvars a b
Expected display: % a and b are removed; c remains defined"####,
                r####"Keep selected variables while clearing the rest
raw = 1;
answer = 42;
tmp = 3;
clearvars -except answer
Expected display: % answer remains defined; raw and tmp are removed"####,
                r####"Clear selected variables while preserving exclusions
a = 1;
b = 2;
c = 3;
untouched = 4;
clearvars a b c -except b
Expected display: % a and c are removed; b and untouched remain defined"####,
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "typed-basic",
        title: "exclude a name",
        program: "x = 1; y = 2; clearvars -except y; z = exist('x','var')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(z == 0); assert(y == 2);",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: r####"How is `clearvars` different from `clear`?"####,
            answer: r####"`clearvars` focuses on workspace variables and supports the `-except` option for preserving named bindings while clearing the rest."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Can I clear only one variable?"####,
            answer: r####"Yes. Use `clearvars x` or `clearvars("x")` to remove one variable, and pass multiple names to clear more than one binding in a single call."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does `clearvars -except x` clear figures or the console?"####,
            answer: r####"No. It only mutates workspace variables. Use `close all` for figures and `clc` for the visible console."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does GPU residency matter?"####,
            answer: r####"No. `clearvars` operates on session bookkeeping, not on array kernels or provider-managed computations."####,
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: r####"clear"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./clear"####),
        },
        BuiltinDocumentationLink {
            label: r####"who"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./who"####),
        },
        BuiltinDocumentationLink {
            label: r####"whos"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./whos"####),
        },
        BuiltinDocumentationLink {
            label: r####"clc"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./clc"####),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: r####"`crates/runmat-runtime/src/builtins/introspection/clearvars.rs`"####,
            target: BuiltinDocumentationLinkTarget::Source(
                r####"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/clearvars.rs"####,
            ),
        },
        BuiltinDocumentationLink { label: "Catalog contract", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/clearvars/contract.rs") },
        BuiltinDocumentationLink { label: "Catalog inference", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/clearvars/inference.rs") }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Legacy test reference",
            location: r####"command_controls::clearvars_command_clears_workspace_state"####,
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
