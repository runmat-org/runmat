use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink { label: "Compiler lowering", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/compiler/parallel.rs") },
        BuiltinDocumentationLink { label: "Future execution", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/interpreter/dispatch/parallel/tasks.rs") },
    ],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Future scheduling, result, cancellation, and completion-order tests", location: "crates/runmat-core/src/tests.rs parallel future tests" }],
    notes: &[],
};

const SCHEDULING_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The scheduler freezes the callable identity, requested output count, arguments, executable bundle, and pool generation into a typed work request. The returned future records its state, outputs, failure, cancellation, and owning generation.",
        "Omitting the pool uses the current pool or creates an automatic one. Worker and transport failures retain structured error identity and source information; program errors are not retried as infrastructure failures.",
    ] },
    BuiltinDocumentationSection { heading: "Execution placement", paragraphs: &[
        "Scheduling does not reinterpret the callable on a worker. The worker executes the compiler-owned function identity and transferred values using the host adapter selected by the pool.",
    ] },
];
const FETCH_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Fetching waits through the session's async execution service and preserves the scheduled callable's output count, classes, shapes, and structured errors. Reading a future does not redefine its result contract.",
        "Future arrays retain one-based source indices. `fetchNext` marks one completed result as read; `fetchOutputs(..., \"UniformOutput\", false)` returns cell arrays when outputs should not be concatenated.",
    ] },
    BuiltinDocumentationSection { heading: "Execution placement", paragraphs: &[
        "Fetch operations cross the worker-to-client result boundary. Numeric placement after transfer follows the scheduled callable and result-assembly contract rather than a provider rule owned by the fetch builtin.",
    ] },
];

const PARFEVAL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "automatic-pool", title: "Schedule work on the current or automatic pool", program: "future = parfeval(@(x) x + 1, 1, 4);\nanswer = fetchOutputs(future)", display_output: Some("answer = 5"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(answer == 5);\ndelete(gcp(\"nocreate\"));" } },
    BuiltinExample { id: "explicit-pool", title: "Schedule work on an explicit pool", program: "pool = parpool(2);\nfuture = parfeval(pool, @(x) x.^2, 1, 5);\nanswer = fetchOutputs(future)", display_output: Some("answer = 25"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(answer == 25);\nassert(future.NumOutputArguments == 1);\ndelete(pool);" } },
];
const PARFEVAL_ON_ALL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "every-worker", title: "Schedule one invocation per worker", program: "pool = parpool(2);\nfuture = parfevalOnAll(pool, @(x) x + 2, 1, 5);\nanswers = fetchOutputs(future)", display_output: Some("answers contains one value 7 per worker"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(answers) == 2);\nassert(all(answers(:) == 7));\ndelete(pool);" } },
];
const FETCH_OUTPUTS_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "multiple-results", title: "Retrieve multiple requested outputs", program: "pool = parpool(2);\nfuture = parfeval(pool, @min, 2, [4 9 2]);\n[value, index] = fetchOutputs(future)", display_output: Some("value = 2; index = 3"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(value == 2);\nassert(index == 3);\ndelete(pool);" } },
    BuiltinExample { id: "cell-output", title: "Return future-array results as cells", program: "first = parfeval(@(x) x + 1, 1, 4);\nsecond = parfeval(@(x) x + 2, 1, 8);\nfutures = first; futures(2) = second;\nanswers = fetchOutputs(futures, \"UniformOutput\", false)", display_output: Some("answers = {5; 10}"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(answers{1}, 5));\nassert(isequal(answers{2}, 10));\ndelete(gcp(\"nocreate\"));" } },
];
const FETCH_NEXT_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "completion-order", title: "Read completed futures one at a time", program: "pool = parpool(2);\nfirst = parfeval(pool, @(x) x + 1, 1, 4);\nsecond = parfeval(pool, @(x) x + 2, 1, 8);\nfutures = first; futures(2) = second;\n[index, value] = fetchNext(futures)", display_output: Some("index identifies the completed future and value is its result"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(index == 1 || index == 2);\nassert(value == 5 || value == 10);\ndelete(pool);" } },
];

const SCHEDULE_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does the output-count argument mean?", answer: "It is the number of outputs requested from the scheduled callable and retained by the future." },
    BuiltinDocumentationFaq { question: "Can I omit the pool?", answer: "Yes. RunMat uses the current compatible pool or creates an automatic pool." },
    BuiltinDocumentationFaq { question: "Can a future outlive its pool?", answer: "No. It is fenced to the pool generation that scheduled it." },
    BuiltinDocumentationFaq { question: "How are worker failures reported?", answer: "Structured identifiers, messages, call stacks, and source locations cross the execution boundary." },
];
const FETCH_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does fetching change output classes?", answer: "No. Output facts come from the scheduled callable and its result contract." },
    BuiltinDocumentationFaq { question: "Can fetching suspend?", answer: "Yes. It waits through the session async runtime until a matching result, timeout, cancellation, or failure is available." },
    BuiltinDocumentationFaq { question: "What does `UniformOutput` control?", answer: "For future arrays, false preserves each result in a cell instead of concatenating compatible values." },
    BuiltinDocumentationFaq { question: "Can the same result be returned twice by `fetchNext`?", answer: "No. A selected future is marked read before the next completion-order selection." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Parallel execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/execution/parallel"),
    },
    BuiltinDocumentationLink {
        label: "parpool",
        target: BuiltinDocumentationLinkTarget::Builtin("parpool"),
    },
    BuiltinDocumentationLink {
        label: "gcp",
        target: BuiltinDocumentationLinkTarget::Builtin("gcp"),
    },
];

macro_rules! future_documentation {
    ($constant:ident, $title:literal, $summary:literal, $description:literal, $keywords:expr, $related:expr, $sections:expr, $examples:expr, $faqs:expr) => {
        pub(in super::super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($title),
            slug: Some($title),
            summary: $summary,
            description: $description,
            keywords: $keywords,
            related: $related,
            sections: $sections,
            examples: $examples,
            example_exemption: None,
            faqs: $faqs,
            links: LINKS,
            media: &[],
            evidence: EVIDENCE,
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Stable),
        };
    };
}

future_documentation!(PARFEVAL_DOCUMENTATION, "parfeval", "Schedule one asynchronous function invocation on a pool.", "`parfeval` freezes a callable, output count, and arguments into a typed work request and returns its future.", &["parallel", "future", "async", "parfeval", "scheduler"], &["fetchNext", "fetchOutputs", "gcp", "parfevalOnAll", "parpool"], SCHEDULING_SECTIONS, PARFEVAL_EXAMPLES, SCHEDULE_FAQS);
future_documentation!(PARFEVAL_ON_ALL_DOCUMENTATION, "parfevalOnAll", "Schedule one function invocation on every worker in a pool.", "`parfevalOnAll` creates one aggregate future for a callable scheduled once on each admitted worker.", &["parallel", "future", "async", "workers", "broadcast"], &["fetchOutputs", "gcp", "parfeval", "parpool"], SCHEDULING_SECTIONS, PARFEVAL_ON_ALL_EXAMPLES, SCHEDULE_FAQS);
future_documentation!(FETCH_OUTPUTS_DOCUMENTATION, "fetchOutputs", "Wait for futures and return their requested outputs.", "`fetchOutputs` retrieves completed outputs without changing the scheduled callable's type, shape, error, or residency contract.", &["parallel", "future", "wait", "outputs", "UniformOutput"], &["fetchNext", "parfeval", "parfevalOnAll", "parpool"], FETCH_SECTIONS, FETCH_OUTPUTS_EXAMPLES, FETCH_FAQS);
future_documentation!(FETCH_NEXT_DOCUMENTATION, "fetchNext", "Return the next completed unread future result.", "`fetchNext` selects a completed unread future by completion order and returns its one-based index followed by requested outputs.", &["parallel", "future", "wait", "completion order", "timeout"], &["fetchOutputs", "parfeval", "parpool"], FETCH_SECTIONS, FETCH_NEXT_EXAMPLES, FETCH_FAQS);
