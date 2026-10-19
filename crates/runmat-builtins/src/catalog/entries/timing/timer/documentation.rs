//! Catalog-owned timer prose and executable examples.

use crate::*;

const TIMER_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "timer-1",
        title: "Start and join a one-shot timer",
        program: "t = timer('TimerFcn', @(src, evt) 0);\nstart(t);\nwait(t);\ntasks = t.TasksExecuted;\nrunning = t.Running;\ndelete(t);\nvalid = isvalid(t)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tasks == 1 && strcmp(running, 'off') && ~valid);",
        },
    },
    BuiltinExample {
        id: "timer-2",
        title: "Find a named timer",
        program: "t = timer('Name', 'batchTimer', 'Tag', 'batch', 'TimerFcn', @(src, evt) 0);\nfound = timerfind('Tag', 'batch');\nfoundName = found.Name;\ndelete(t)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(foundName, 'batchTimer'));",
        },
    },
];

const TIMERFIND_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "timerfind-1",
    title: "Find only visible timers",
    program: "visible = timer('Name', 'visibleTimer', 'Tag', 'group', 'TimerFcn', @(src, evt) 0);\nhidden = timer('Name', 'hiddenTimer', 'Tag', 'group', 'ObjectVisibility', 'off', 'TimerFcn', @(src, evt) 0);\nfound = timerfind('Tag', 'group');\nname = found.Name;\ndelete(visible);\ndelete(hidden)",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(name, 'visibleTimer'));",
    },
}];

const TIMERFINDALL_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "timerfindall-1",
    title: "Include invisible timers",
    program: "t = timer('Name', 'hiddenTimer', 'ObjectVisibility', 'off', 'TimerFcn', @(src, evt) 0);\nallTimers = timerfindall('Name', 'hiddenTimer');\nname = allTimers.Name;\ndelete(t)",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(name, 'hiddenTimer'));",
    },
}];

const TIMER_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink { label: "timer runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/timer.rs") },
        BuiltinDocumentationLink { label: "timer lifecycle", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/timer/lifecycle.rs") },
        BuiltinDocumentationLink { label: "timer callbacks", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/timer/callback.rs") },
    ],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "session timer ownership and callback tests", location: "crates/runmat-runtime/src/builtins/timing/timer/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM timer execution", location: "crates/runmat-core/src/tests/timers.rs:timer_constructs_finds_starts_and_deletes_through_vm" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::BrowserTest, label: "browser timer task lifecycle", location: "crates/runmat-runtime/src/builtins/timing/timer/browser_tests.rs" },
    ],
    notes: &["Text callbacks require an executor-owned base-workspace source-evaluation producer, which is not yet installed. Source declarations and registered browser symbols are not execution evidence."],
};

pub(super) const TIMER: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("timer"), slug: Some("timer"),
    summary: "Create a session-owned timer handle for scheduled callbacks.",
    description: "`timer` creates a handle object with callback, timing, execution, visibility, and user-data properties. `start`, `startat`, `stop`, `wait`, and `delete` manage its lifecycle.",
    keywords: &["timer", "callback", "start", "stop", "wait", "timerfind"],
    related: &["timerfind", "timerfindall", "pause"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The default timer is single-shot with Period=1 second, StartDelay=0, TasksToExecute=1, BusyMode='drop', ObjectVisibility='on', Running='off', TasksExecuted=0, and a generated Name.",
        "Callback properties accept text, a function handle or closure, or a callback cell. Handle callbacks invoke the session callable service. Nonempty text callbacks require executor-owned source evaluation in the live base workspace; without that producer start fails before Running becomes on.",
        "start returns after scheduling, while the host's local task driver polls callbacks. wait joins without stopping. stop cancels and invokes StopFcn once; delete invalidates the handle and removes it from discovery.",
        "ErrorFcn receives the original callback error in event Data.Error; StopFcn runs once after a failing generation, and wait reports the original error.",
        "Fixed-rate, fixed-delay, fixed-spacing, and single-shot modes use the shared monotonic timer and local executor. The timer itself has no GPU kernel or fusion path; admitted numeric resident properties materialize through their owning provider.",
        "MATLAB start, stop, wait, and delete methods do not return a value. RunMat's assignable numeric status is an explicitly gated extension. startat accepts an absolute datetime, date text, date vector, serial-date array, or three/six calendar components within 25 days; omitted time and scalar relative seconds remain separately gated RunMat extensions pending the scalar serial-date compatibility decision. Timezone-bearing datetime, relative date words, and some MATLAB date-string formats remain unsupported.",
    ] }],
    examples: TIMER_EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does start wait for callbacks?", answer: "No. It schedules the timer; call wait to join the task. The session host must drive local tasks while idle." },
        BuiltinDocumentationFaq { question: "Can callbacks use GPU arrays?", answer: "Yes, when the invoked callback uses GPU-capable builtins. The timer object itself has no provider kernel." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "timerfind", target: BuiltinDocumentationLinkTarget::Builtin("timerfind") },
        BuiltinDocumentationLink { label: "timerfindall", target: BuiltinDocumentationLinkTarget::Builtin("timerfindall") },
        BuiltinDocumentationLink { label: "pause", target: BuiltinDocumentationLinkTarget::Builtin("pause") },
    ],
    media: &[], evidence: TIMER_EVIDENCE,
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const TIMERFIND: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("timerfind"), slug: Some("timerfind"),
    summary: "Find visible, valid timers in the current session.",
    description: "`timerfind` returns visible valid timer handles whose properties match optional name-value filters. A single match is scalar; zero or multiple matches form a cell row.",
    keywords: &["timerfind", "timer", "visibility"], related: &["timer", "timerfindall"],
    sections: &[BuiltinDocumentationSection { heading: "Matching", paragraphs: &[
        "Property names decode case-insensitively. Character rows normalize to strings; native integer filters compare exactly, including values outside binary64's exact range. Numeric resident filters materialize through their owner.",
        "Deleted and invisible timers are omitted. Use timerfindall to include invisible timers.",
    ] }],
    examples: TIMERFIND_EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Which search forms are available?", answer: "No-argument and name-value searches are available. Searching within a supplied timer array and filtering from a property structure are not yet implemented." }],
    links: &[
        BuiltinDocumentationLink { label: "timer", target: BuiltinDocumentationLinkTarget::Builtin("timer") },
        BuiltinDocumentationLink { label: "timerfindall", target: BuiltinDocumentationLinkTarget::Builtin("timerfindall") },
    ],
    media: &[], evidence: TIMER_EVIDENCE,
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const TIMERFINDALL: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("timerfindall"), slug: Some("timerfindall"),
    summary: "Find valid timers including invisible timers.",
    description: "`timerfindall` searches the current session's timer registry regardless of ObjectVisibility, with optional exact name-value filters.",
    keywords: &["timerfindall", "timer", "visibility"], related: &["timer", "timerfind"],
    sections: &[BuiltinDocumentationSection { heading: "Matching", paragraphs: &[
        "The same property and exact integer-filter semantics as timerfind apply; ObjectVisibility does not exclude a match.",
    ] }],
    examples: TIMERFINDALL_EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Which search forms are available?", answer: "No-argument and name-value searches are available. Searching within a supplied timer array and filtering from a property structure are not yet implemented." }],
    links: &[
        BuiltinDocumentationLink { label: "timer", target: BuiltinDocumentationLinkTarget::Builtin("timer") },
        BuiltinDocumentationLink { label: "timerfind", target: BuiltinDocumentationLinkTarget::Builtin("timerfind") },
    ],
    media: &[], evidence: TIMER_EVIDENCE,
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const INTERNAL_METHOD: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: None, slug: None,
    summary: "Internal timer method binding.",
    description: "The compiler routes timer handle methods through these hidden, session-bound runtime identities. MATLAB method calls have no output; assignable numeric status and startat relative seconds are separately gated RunMat-only forms. Absolute startat dates use the MATLAB method identity. User-facing syntax and examples belong to timer.",
    keywords: &["timer", "method"], related: &["timer"], sections: &[],
    examples: &[], example_exemption: Some("Hidden method bindings are exercised by timer lifecycle examples and VM method tests."),
    faqs: &[], links: &[], media: &[], evidence: TIMER_EVIDENCE,
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const INTERNAL_SETTER: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: None, slug: None,
    summary: "Internal timer dependent-property setter.",
    description: "The object runtime routes dependent timer property assignments through these method-only validation bindings. User-facing property forms and examples belong to timer.",
    keywords: &["timer", "property"], related: &["timer"], sections: &[],
    examples: &[], example_exemption: Some("Method-only setters are exercised by timer property examples and VM assignment tests."),
    faqs: &[], links: &[], media: &[], evidence: TIMER_EVIDENCE,
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
