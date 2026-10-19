use crate::*;

const ERROR_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "error-1",
        title: "Throw a formatted error with an identifier",
        program:
            "x = 1;\nerror('MyComponent:incorrectType','Input must be a char, not a %s.',class(x))",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "MyComponent:incorrectType",
        },
    },
    BuiltinExample {
        id: "error-2",
        title: "Format an exact typed integer",
        program: "error('Value is %u',uint64(18446744073709551615))",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:error",
        },
    },
];

pub(super) const ERROR: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("error"), slug: Some("error"),
    summary: "Throw an error with optional identifier and formatted message values.", description: "`error` constructs a diagnostic message and transfers control to the nearest compatible catch boundary. A one-input message is literal; multi-input forms decode documented escapes and apply sprintf-style formatting before throwing.",
    keywords: &["error", "exception", "identifier", "formatting", "try catch"], related: &["sprintf", "warning"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior and placement",
        paragraphs: &["`error(msg)` throws the literal text scalar without converting escape or format sequences.",
        "`error(msg,A)` and `error(errID,___)` decode escape sequences, format replacement values, and then throw.",
        "Documented error identifiers use colon-separated fields that each begin with a letter and otherwise contain only letters, digits, or underscores.",
        "All eight integer classes are documented numeric formatting values. RunMat keeps typed integers authoritative through integer conversion specifiers, including `uint64` values above binary64's exact range.",
        "If every supplied input is an empty array, the call is a no-op and does not throw.",
        "`error(MException)`, unqualified identifier normalization, and the `messageid` or `msg` structure aliases are independent RunMat-only extensions and reject in MATLAB compatibility mode.",
        "The documented matlab.lang.correction object families are not yet representable in RunMat and currently raise an explicit unsupported diagnostic.",
        "An errorStruct must be scalar and contain at least one of `message`, `identifier`, or `stack`. RunMat accepts `message` and `identifier` independently; explicit `stack` reconstruction is deferred because RuntimeError does not yet accept MATLAB stack structures.",
        "Formatting values gather to host after message and identifier validation; error is a host control-flow operation.",
        "The legacy examples' prose outcomes describe the thrown diagnostic; error calls do not print a successful display result."] }],
    examples: ERROR_EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does error execute on the GPU?", answer: "No. MathWorks documents GPU-array inputs, but the function does not run on the GPU. RunMat gathers valid formatting values and throws from the host." }, BuiltinDocumentationFaq { question: "Are suggested corrections implemented?", answer: "Not yet. RunMat does not currently model the three matlab.lang.correction object classes, so correction-object calls report the infrastructure gap explicitly." }, BuiltinDocumentationFaq { question: "Does errorStruct preserve an explicit stack?", answer: "Not yet. Message and identifier fields are supported, but translating MATLAB stack structure arrays into RuntimeError call frames remains deferred." }],
    links: &[BuiltinDocumentationLink { label: "MathWorks error documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/error.html") }, BuiltinDocumentationLink { label: "sprintf", target: BuiltinDocumentationLinkTarget::Builtin("sprintf") }, BuiltinDocumentationLink { label: "warning", target: BuiltinDocumentationLinkTarget::Builtin("warning") }],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime error", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/diagnostics/error.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "error runtime semantics", location: "crates/runmat-runtime/src/builtins/diagnostics/error.rs tests" }], notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

const RETHROW_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "rethrow-1", title: "Propagate an exception to an outer catch block", program: "try\n    try\n        error(\"RunMat:demo:badInput\", \"Bad input.\");\n    catch err\n        rethrow(err);\n    end\ncatch outer\n    caughtID = outer.identifier;\n    disp(caughtID)\nend", display_output: Some("RunMat:demo:badInput"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(caughtID, 'RunMat:demo:badInput'));" } },
    BuiltinExample { id: "rethrow-2", title: "Use the RunMat-only no-argument shorthand inside a catch block", program: "try\n    try\n        error(\"RunMat:oops\", \"x\");\n    catch\n        rethrow();\n    end\ncatch err\n    caughtMessage = err.message;\n    disp(caughtMessage)\nend", display_output: Some("x"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(caughtMessage, 'x'));" } },
];

pub(super) const RETHROW: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("rethrow"), slug: Some("rethrow"),
    summary: "Rethrow a caught exception so it propagates to an outer handler.", description: "`rethrow` re-raises an exception so an outer `catch` block or embedding runtime can handle it. Passing an `MException` preserves identifier and message, while passing text creates a new error payload; the RunMat-only `rethrow()` form rethrows the active lexical catch exception.",
    keywords: &["rethrow", "exception", "catch", "error propagation"], related: &["error", "warning", "assert"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior and placement",
        paragraphs: &["`rethrow(err)` preserves the identifier and message of an `MException` value.",
        "The `identifier` and `message` properties are character vectors through both dot access and `getfield`; empty text is a 0-by-0 character array.",
        "A caught `MException` retains the original runtime span, call frames, phase, failure policy, and owning typed source across rethrow; a manually constructed `MException` has only its public identifier, message, and stack.",
        "`rethrow(message)` is a RunMat-only extension that raises a fresh error using the supplied string as the message.",
        "Other value types produce a generic `RunMat:error`-style failure describing the value.",
        "`rethrow` never returns normally; it immediately transfers control outward to the next enclosing error handler.",
        "In RunMat `catch` blocks, `rethrow()` with no argument uses that active catch's exception. A callback invoked from an active catch can also use it; after the catch exits, an earlier exception is not inherited.",
        "Integer semantics: `rethrow` consumes an exception object and has no numeric data, control, output-class, or provider role; numeric and resident values fail directly without gather or provider access.",
        "`rethrow` is control flow only and never executes on the GPU.",
        "`rethrow` performs no provider dispatch; it raises or propagates a host-side runtime error immediately."] }],
    examples: RETHROW_EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "When should I use `rethrow` instead of `error`?", answer: "Use `rethrow` when you want to propagate the current exception rather than constructing a new one from scratch." }, BuiltinDocumentationFaq { question: "Does `rethrow(err)` preserve the original identifier?", answer: "Yes. When `err` is an `MException`, RunMat preserves both the identifier and the message." }, BuiltinDocumentationFaq { question: "Is `rethrow()` without an argument supported?", answer: "Yes, as a RunMat extension while a catch is active, including in a callback it invokes. It fails once that catch has exited." }],
    links: &[BuiltinDocumentationLink { label: "error", target: BuiltinDocumentationLinkTarget::Builtin("error") }, BuiltinDocumentationLink { label: "warning", target: BuiltinDocumentationLinkTarget::Builtin("warning") }, BuiltinDocumentationLink { label: "assert", target: BuiltinDocumentationLinkTarget::Builtin("assert") }],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime rethrow", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/diagnostics/rethrow.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "rethrow runtime semantics", location: "crates/runmat-runtime/src/builtins/diagnostics/rethrow.rs tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM nested catch and rethrow", location: "crates/runmat-vm/tests/exceptions.rs::nested_try_catch_rethrow" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Forced native nested catch and rethrow", location: "crates/runmat-core/src/tests.rs::forced_generic_native_executable_lane_matches_structured_try_catch_transfer" }], notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
