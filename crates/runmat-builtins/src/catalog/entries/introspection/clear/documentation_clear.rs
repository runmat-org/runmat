use crate::*;

pub(super) const CLEAR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"clear"####),
    slug: Some(r####"clear"####),
    summary: r####"Clear variables from the active workspace in MATLAB and RunMat."####,
    description: r####"`clear` removes variables from the active workspace. It supports `clear`, `clear all`, and named-variable forms (for example `clear x`), following MATLAB and RunMat workspace-clearing semantics."####,
    keywords: &[
        r####"clear"####,
        r####"workspace"####,
        r####"variables"####,
        r####"clear all"####,
        r####"command window"####,
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and placement",
            paragraphs: &[
                r####"`clear` with no inputs clears all workspace variables created in the current session."####,
                r####"Bare `clear` removes workspace variables. `clear('all')` and command-form `clear all` additionally request dynamic-function cache invalidation when no workspace variable named `all` shadows the directive."####,
                r####"`clear x` removes the variable `x` while leaving other workspace bindings intact."####,
                r####"Multiple names can be supplied as separate inputs or command-form tokens, for example `clear x y z`."####,
                r####"Active `onCleanup` objects are invoked before their workspace bindings are removed."####,
                r####"After a successful clear, removed variables are no longer available to later statements and the workspace snapshot is refreshed so hosts can drop cleared bindings."####,
                r####"Accepts variable names as string scalars, row character vectors, or string arrays. A matching workspace binding takes precedence over an option-like name; otherwise `all`, `variables`, `functions`, and `mex` select their respective clear directives."####,
                r####"Clearing a name that does not exist is a no-op."####,
                r####"`clear` is a sink builtin and does not produce a meaningful output value."####,
                r####"GPU support: Runs on the host CPU because it mutates the session workspace rather than array contents."####,
                r####"`clear` performs host-side workspace mutation only. It does not launch GPU kernels, does not participate in fusion, and does not depend on acceleration providers."####,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Legacy examples and displayed results",
            paragraphs: &[
                r####"Clear the entire interactive workspace
x = 1;
y = magic(3);
clear
Expected display: % x and y are removed from the workspace"####,
                r####"Use MATLAB-style command form
a = 42;
clear all
Expected display: % a is removed from the workspace"####,
                r####"Accessing a cleared variable raises an error
x = 7;
clear;
x
Expected display: Undefined variable: x"####,
                r####"Clear one variable while leaving others intact
x = 1;
y = 2;
clear x
Expected display: % x is removed from the workspace while y remains defined"####,
                r####"Clear multiple named variables in command form
a = 1;
b = 2;
c = 3;
clear a b
Expected display: % a and b are removed; c remains defined"####,
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "typed-basic",
        title: "clear one name",
        program: "x = 1; y = 2; clear x; z = exist('x','var')",
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
            question: r####"Does `clear` remove every variable in the current session?"####,
            answer: r####"Bare `clear` clears workspace variables. `clear all` also requests dynamic-function cache invalidation unless a workspace variable named `all` shadows the directive. Named forms such as `clear x` remove only the specified binding."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Can I clear only one variable?"####,
            answer: r####"Yes. Use `clear x` or `clear("x")` to remove one variable, and pass multiple names to clear more than one binding in a single call."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does `clear` return anything?"####,
            answer: r####"No meaningful value is returned. `clear` is a side-effecting builtin whose purpose is to mutate workspace state."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does `clear` affect plotting state or the console?"####,
            answer: r####"It does not close figures or clear the visible console. Bare `clear` removes workspace variables; `clear all` can also invalidate dynamic functions. Use `close all` for figures and `clc` for the console."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does GPU residency matter?"####,
            answer: r####"No. `clear` operates on session bookkeeping, not on array kernels or provider-managed computations."####,
        },
    ],
    links: &[
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
        BuiltinDocumentationLink {
            label: r####"close"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./close"####),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: r####"`crates/runmat-runtime/src/builtins/introspection/clear.rs`"####,
            target: BuiltinDocumentationLinkTarget::Source(
                r####"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/clear.rs"####,
            ),
        },
        BuiltinDocumentationLink { label: "Catalog contract", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/clear/contract.rs") },
        BuiltinDocumentationLink { label: "Catalog inference", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/clear/inference.rs") }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Legacy test reference",
            location: r####"command_controls::clear_command_clears_workspace_state"####,
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
