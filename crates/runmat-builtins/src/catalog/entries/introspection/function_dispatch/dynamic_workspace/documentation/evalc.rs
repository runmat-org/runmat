use crate::*;

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("evalc"),
    slug: Some("evalc"),
    summary: "Evaluate source text in the current workspace and capture command-window output.",
    description: "`evalc` evaluates scalar source text in the current workspace through the active dynamic-source execution service. Its first output is captured command-window text; later requested outputs receive values produced by the source.",
    keywords: &["evalc", "eval", "capture", "command window", "dynamic workspace"],
    related: &["eval", "evalin"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`results = evalc(expression)` captures command-window stdout and stderr instead of forwarding them to the normal execution stream. RunMat currently represents the first output as a scalar string; MATLAB's character-vector representation is not implied by that RunMat storage choice. Silent source returns empty text.",
                "`[results,varargout] = evalc(expression)` forwards additional values produced by the source. Their count, class, shape, and residency are dynamic, and the source may read or update the current workspace.",
                "The argument must be scalar string-like source text. Successful assignments become visible to subsequent requests and dynamic workspace evaluations. A dynamically created name is not by itself a static proof that every already-lowered reference exists.",
                "An error from evaluated source propagates; its uncommitted dynamic workspace completion is not published. An active workspace execution service is required.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Capture and diary",
            paragraphs: &[
                "Captured command-window text is returned rather than emitted to the request stream. The active capture scope bypasses normal diary recording for those entries.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "`evalc` is a host-side execution and capture boundary: it launches no numeric GPU kernel, performs no numeric gather, and does not participate in fusion itself. Evaluated source may call accelerated builtins under their own contracts.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "capture-display",
            title: "Capture displayed text",
            program: "captured = evalc('disp(42)');",
            display_output: Some("captured contains 42 followed by a newline"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(contains(captured, '42'));" },
        },
        BuiltinExample {
            id: "capture-and-result",
            title: "Capture output and return an evaluated value",
            program: "[captured, value] = evalc('disp(5); 6 * 7');",
            display_output: Some("captured contains 5; value = 42"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(contains(captured, '5')); assert(value == 42);" },
        },
        BuiltinExample {
            id: "persist-assignment",
            title: "Read an assignment through a later dynamic evaluation",
            program: "captured = evalc('scale = 3; disp(scale)'); observed = eval('scale');",
            display_output: Some("captured contains 3; observed = 3"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(contains(captured, '3')); assert(observed == 3);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `evalc` print captured text to stdout?", answer: "No. Captured command-window stdout and stderr are returned as the first output instead of being forwarded to the normal stream." },
        BuiltinDocumentationFaq { question: "Does `evalc` write captured text to `diary`?", answer: "No. Entries intercepted by the active capture scope bypass normal diary recording." },
        BuiltinDocumentationFaq { question: "Where does `evalc` execute source text?", answer: "In the current workspace through the active dynamic-source execution service; it is not limited to a VM implementation when another admitted service supplies that authority." },
        BuiltinDocumentationFaq { question: "Can later statements use names created by `evalc`?", answer: "Successful assignments persist. A dynamic write is not a static existence proof for arbitrary already-lowered references; dynamic evaluation or explicit initialization can read the resulting binding under its normal lookup rules." },
        BuiltinDocumentationFaq { question: "Can `evalc` return values besides captured text?", answer: "Yes. With multiple requested outputs, later outputs receive values produced by the evaluated source." },
        BuiltinDocumentationFaq { question: "Does `evalc` have a GPU fast path?", answer: "No. It is an execution and capture boundary. The evaluated source can still call accelerated builtins." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "diary", target: BuiltinDocumentationLinkTarget::Builtin("diary") },
        BuiltinDocumentationLink { label: "disp", target: BuiltinDocumentationLinkTarget::Builtin("disp") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime dynamic-workspace binding", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/dynamic_workspace.rs") },
            BuiltinDocumentationLink { label: "VM evalc dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/call/builtins.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dynamic-workspace fallback", location: "builtins::introspection::dynamic_workspace::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Capture, stream isolation, silent source, and persistence", location: "dynamic_workspace_evalc_captures_console_without_stream_leak; dynamic_workspace_evalc_returns_empty_capture_for_silent_source" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
