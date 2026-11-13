use crate::*;

const TIC_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "tic-1", title: "Measuring a simple loop", program: "tic;\nfor k = 1:1e5\n    sqrt(k);\nend\nelapsed = toc", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isfinite(elapsed) && elapsed >= 0);" } },
    BuiltinExample { id: "tic-2", title: "Capturing and reusing the tic handle", program: "t = tic;\nx = sum(1:1000);\nelapsed = toc(t)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(t, 'uint64') && elapsed >= 0);" } },
    BuiltinExample { id: "tic-3", title: "Nesting timers for staged profiling", program: "outer = tic;\nx = sum(1:1000);\ninner = tic;\ny = sum(1:2000);\ninnerT = toc(inner);\nouterT = toc(outer)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(innerT >= 0 && outerT >= innerT);" } },
    BuiltinExample { id: "tic-4", title: "Measuring a suspending wait", program: "token = tic;\npause(0.01);\nelapsed = toc(token)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(elapsed >= 0);" } },
    BuiltinExample { id: "tic-5", title: "Resetting the stopwatch after a measurement", program: "elapsed1 = toc(tic);\npause(0.1);\nelapsed2 = toc(tic)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(elapsed1 >= 0 && elapsed2 >= 0);" } },
];

const TOC_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "toc-1", title: "Measuring elapsed time since the last tic", program: "tic;\npause(0.25);\nelapsed = toc", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(elapsed >= 0);" } },
    BuiltinExample { id: "toc-2", title: "Using toc with an explicit tic handle", program: "token = tic;\nx = sum(1:1000);\nelapsed = toc(token)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(elapsed >= 0);" } },
    BuiltinExample { id: "toc-3", title: "Timing nested stages with toc", program: "outer = tic;\nx = sum(1:1000);\ninner = tic;\ny = sum(1:2000);\nstage2Time = toc(inner);\ntotalTime = toc(outer)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(totalTime >= stage2Time && stage2Time >= 0);" } },
    BuiltinExample { id: "toc-4", title: "Printing elapsed time without capturing output", program: "tic;\nx = sum(1:1000);\ntoc;", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "elapsed = toc; assert(elapsed >= 0);" } },
    BuiltinExample { id: "toc-5", title: "Measuring immediately with toc(tic)", program: "elapsed = toc(tic);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(elapsed >= 0);" } },
];

pub(super) const TIC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("tic"), slug: Some("tic"),
    summary: "Start a high-resolution session-owned stopwatch.",
    description: "`tic` starts a monotonic wall-clock stopwatch. Capture its opaque `uint64` token when a later `toc(token)` must refer to that start; an uncaptured `tic` only updates the latest start.",
    keywords: &["tic", "timing", "profiling", "benchmark"], related: &["toc", "timeit", "cputime", "pause"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The returned uint64 is an opaque session token, not an encoded timestamp. A token from another session or execution generation is invalid.",
        "`tic` without an observed output updates only the latest start; repeated statement-form starts do not retain unreachable handles.",
        "`tic` uses the invocation's monotonic timing service and never enters GPU fusion or provider execution.",
    ] }], examples: TIC_EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `tic` print anything when called without a semicolon?", answer: "No. `tic` is a sink builtin; the stopwatch token is displayed only when explicitly requested." },
        BuiltinDocumentationFaq { question: "Is the returned handle portable across sessions?", answer: "No. The opaque token is valid only in its owning RunMat session and cannot be saved for a later session." },
        BuiltinDocumentationFaq { question: "Can I run `tic` on a worker thread?", answer: "Yes. Independent runtime contexts own independent stopwatch state; retain explicit tokens when starts may interleave within a context." },
        BuiltinDocumentationFaq { question: "How accurate is the measurement?", answer: "The monotonic host clock determines resolution; RunMat adds no artificial jitter or throttling." },
        BuiltinDocumentationFaq { question: "Does `tic` participate in GPU fusion?", answer: "No. It is a host-side timing boundary." },
    ],
    links: &[BuiltinDocumentationLink { label: "toc", target: BuiltinDocumentationLinkTarget::Builtin("toc") }, BuiltinDocumentationLink { label: "timeit", target: BuiltinDocumentationLinkTarget::Builtin("timeit") }],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Session stopwatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/tic.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "tic session token", location: "crates/runmat-runtime/src/builtins/timing/tic.rs tests" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const TOC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("toc"), slug: Some("toc"),
    summary: "Read elapsed seconds since a session-owned tic start.",
    description: "`toc` returns elapsed wall-clock seconds since the latest `tic` or since an explicit opaque `uint64` token from this session. Called without an output destination, it prints the elapsed-time line instead of an implicit `ans` value.",
    keywords: &["toc", "timing", "profiling", "benchmark"], related: &["tic", "timeit", "cputime"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`toc()` reads the latest start without consuming it. `toc(token)` reads the start associated with a published token; repeated reads remain valid.",
        "Malformed, forged, stale, foreign-session, and legacy double-valued timer handles raise `RunMat:toc:InvalidTimerHandle` before clock access.",
        "The elapsed result is a scalar double in seconds. The monotonic timing service controls resolution; no GPU provider or fusion stage is involved.",
    ] }], examples: TOC_EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What happens if I call `toc` before `tic`?", answer: "It raises `RunMat:toc:NoMatchingTic` when this session has no latest start." },
        BuiltinDocumentationFaq { question: "Does `toc` remove the matching `tic`?", answer: "No. Both forms are read operations, so repeated calls report elapsed time from the same start." },
        BuiltinDocumentationFaq { question: "Can I reuse a `tic` handle after calling `toc(t)`?", answer: "Yes, within the same session. The token remains valid until that session ends." },
        BuiltinDocumentationFaq { question: "Does `toc` print output?", answer: "A standalone `toc` or `toc;` prints an `Elapsed time is ... seconds.` line. Assigning its result returns a scalar double without that line; a semicolon on the standalone call does not suppress its explicit print." },
        BuiltinDocumentationFaq { question: "Is `toc` affected by GPU execution or fusion?", answer: "No. It reads the host monotonic timing service." },
        BuiltinDocumentationFaq { question: "How accurate is the reported time?", answer: "Resolution depends on the host monotonic clock and scheduler." },
    ],
    links: &[BuiltinDocumentationLink { label: "tic", target: BuiltinDocumentationLinkTarget::Builtin("tic") }, BuiltinDocumentationLink { label: "timeit", target: BuiltinDocumentationLinkTarget::Builtin("timeit") }],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Session stopwatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/toc.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "toc session token", location: "crates/runmat-runtime/src/builtins/timing/toc.rs tests" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
