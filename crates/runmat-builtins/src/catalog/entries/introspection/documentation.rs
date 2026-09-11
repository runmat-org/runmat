use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/feval.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Callable dispatch and typed forwarding tests", location: "builtins::introspection::feval::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Function and closure dispatch tests", location: "crates/runmat-vm/tests/functions.rs and crates/runmat-vm/tests/closures.rs" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`feval(fun, x1, ..., xN)` invokes a callable selected at runtime. Plain character-vector or string-scalar names use the same resolution path as direct named calls. Function handles use their bound identity, and closures retain captured values.",
        "Arguments are forwarded without numeric conversion. Output count, class, shape, errors, overflow behavior, side effects, and residency belong to the selected callable and the number of outputs requested at the call site.",
        "RunMat mode additionally accepts text targets prefixed with `@` and supported callable object receivers. A MATLAB compatibility pin rejects those extensions before dispatch; ordinary names and actual function handles remain compatible.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "`feval` performs no provider operation of its own. The selected callable determines whether resident inputs remain on their owner, launch a kernel, use a typed fallback, gather, or are rejected.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "plain-name",
        title: "Call a function by its plain name",
        program: "y = feval(\"round\", pi, 2)",
        display_output: Some("y = 3.14"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 3.14) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "function-handle",
        title: "Call a bound function handle",
        program: "f = @max;\ny = feval(f, int32([2 7 4]))",
        display_output: Some("y = int32(7)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, \"int32\"));\nassert(y == int32(7));",
        },
    },
    BuiltinExample {
        id: "multiple-outputs",
        title: "Forward the requested output count",
        program: "[value, index] = feval(@max, uint64([4 9 2]))",
        display_output: Some("value = uint64(9); index = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isa(value, \"uint64\"));\nassert(value == uint64(9));\nassert(index == 2);",
        },
    },
    BuiltinExample {
        id: "closure",
        title: "Invoke a closure with captured values",
        program: "offset = 3;\nf = @(x) x.^2 + offset;\ny = feval(f, [1 2 3])",
        display_output: Some("y = [4 7 12]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [4 7 12]));",
        },
    },
    BuiltinExample {
        id: "gpu-dispatch",
        title: "Dispatch to a provider-capable function",
        program: "G = gpuArray([0 pi/2 pi]);\nH = feval(@sin, G);\nhost = gather(H)",
        display_output: Some("host = [0 1 0] within floating tolerance"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(host - [0 1 0])) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "at-prefixed-text-extension",
        title: "Use an @-prefixed text target in RunMat mode",
        program: "y = feval(\"@sin\", pi / 2)",
        display_output: Some("y = 1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 1) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `feval` convert integer arguments to double?", answer: "No. It forwards each value unchanged; conversion and output-class rules belong to the selected callable." },
    BuiltinDocumentationFaq { question: "Does `feval` gather a gpuArray?", answer: "Not by itself. The selected callable's provider contract controls residency and fallback." },
    BuiltinDocumentationFaq { question: "Can I pass a plain function name?", answer: "Yes. Character-vector and scalar-string names are compatible callable selectors." },
    BuiltinDocumentationFaq { question: "Is an actual function handle compatible?", answer: "Yes. Values such as `@sin` invoke their bound callable identity." },
    BuiltinDocumentationFaq { question: "Is an @-prefixed string portable?", answer: "No. Text such as `\"@sin\"` is a RunMat extension; use `\"sin\"` or `@sin` for compatible source." },
    BuiltinDocumentationFaq { question: "Why can output facts be dynamic?", answer: "The target and requested output count can be known only at runtime. When the target is statically known, the compiler can reuse its catalog contract." },
];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "str2func",
    target: BuiltinDocumentationLinkTarget::Builtin("str2func"),
}];

pub(super) const FEVAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("feval"), slug: Some("feval"), summary: "Invoke a function selected by name, handle, closure, or supported receiver.",
    description: "`feval` dynamically resolves a callable and forwards arguments and requested outputs without imposing its own numeric or provider semantics.",
    keywords: &["feval", "function handle", "dynamic dispatch", "callback", "varargin", "varargout", "closure"],
    related: &["str2func"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
