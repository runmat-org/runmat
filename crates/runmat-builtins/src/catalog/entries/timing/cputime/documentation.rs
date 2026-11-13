use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "cputime-1",
    title: "Measure CPU Time Used By Code",
    program: "tStart = cputime;\nfor k = 1:100000\n    sqrt(k);\nend\ntElapsed = cputime - tStart",
    display_output: Some("tElapsed = <seconds>"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isfinite(tElapsed) && tElapsed >= 0);",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cputime"), slug: Some("cputime"),
    summary: "Return total CPU time used by the current RunMat process.",
    description: "`cputime` reports native process CPU seconds. It is a legacy MATLAB-compatible timer; prefer `timeit` or `tic`/`toc` for wall-clock benchmarking. Browser hosts do not expose process CPU accounting, so RunMat raises `RunMat:cputime:Unavailable` instead of reporting wall time as CPU time.",
    keywords: &["cputime", "cpu time", "timing", "profiling", "legacy"],
    related: &["tic", "toc", "timeit"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`t = cputime` takes no inputs and returns a nonnegative scalar double in seconds on native hosts.",
        "Native Unix uses process CPU accounting via `getrusage`; native Windows uses `GetProcessTimes`. Values accumulate since the process started and do not reset.",
        "Browser/WASM does not expose process CPU accounting and raises `RunMat:cputime:Unavailable`; no wall-clock fallback is mislabeled as CPU time.",
        "`cputime` runs on the host, never queries GPU providers, and is excluded from fusion.",
    ] }],
    examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is `cputime` different from `tic` and `toc`?", answer: "`cputime` measures native process CPU usage; `tic`/`toc` measure elapsed wall-clock time." },
        BuiltinDocumentationFaq { question: "Why is `cputime` unavailable in browsers?", answer: "Browsers do not expose process CPU accounting. A wall-clock reading would have different semantics, so RunMat reports that the capability is unavailable." },
        BuiltinDocumentationFaq { question: "Does `cputime` reset?", answer: "No. Native builds return total process CPU time since process start; subtract two readings to measure a section." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tic", target: BuiltinDocumentationLinkTarget::Builtin("tic") },
        BuiltinDocumentationLink { label: "toc", target: BuiltinDocumentationLinkTarget::Builtin("toc") },
        BuiltinDocumentationLink { label: "timeit", target: BuiltinDocumentationLinkTarget::Builtin("timeit") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime CPU time", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/cputime.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "cputime process and browser capability", location: "crates/runmat-runtime/src/builtins/timing/cputime.rs tests" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
