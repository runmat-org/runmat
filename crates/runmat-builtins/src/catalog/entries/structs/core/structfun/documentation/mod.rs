mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Field-wise callback execution", paragraphs: &[
        "`structfun(func,S)` invokes `func` once for each field of scalar structure `S`, in field order. The input structure is not modified.",
        "Function handles, anonymous and captured functions, and ordinary function names use RunMat's normal callable-resolution rules. Struct arrays are rejected rather than flattened implicitly.",
    ] },
    BuiltinDocumentationSection { heading: "Outputs and error handling", paragraphs: &[
        "`UniformOutput` defaults to true. Each requested callback output must be a compatible scalar; `structfun` collects it into an N-by-1 array, where N is the field count. Fixed-width integers, single values, and supported complex classes retain their native storage when every result has a compatible class.",
        "With `'UniformOutput',false`, each requested result is a scalar structure with the original field names. This form admits heterogeneous callback values and sizes.",
        "When the caller requests several outputs, every callback receives the same requested count and each output position is collected independently. `ErrorHandler` receives a structure containing `identifier`, `message`, `index`, and `field`, followed by the field value that failed.",
    ] },
    BuiltinDocumentationSection { heading: "Provider-resident fields", paragraphs: &[
        "`structfun` controls iteration on the host. A resident field value is gathered through its owning provider before callback invocation, and collected uniform outputs are host-resident. Callback execution makes `structfun` a fusion boundary.",
    ] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does structfun modify the input structure?", answer: "No. It reads each field and creates new collected outputs." },
    BuiltinDocumentationFaq { question: "Can callbacks return strings or structures?", answer: "Yes with `'UniformOutput',false`; each result is stored under the corresponding input field name." },
    BuiltinDocumentationFaq { question: "Can structfun collect multiple callback outputs?", answer: "Yes. Each requested callback output is collected independently and returned in the same output position." },
    BuiltinDocumentationFaq { question: "What happens when the structure has no fields?", answer: "Uniform mode returns an empty 0-by-1 double array for each requested output. Nonuniform mode returns an empty scalar structure." },
    BuiltinDocumentationFaq { question: "Are resident field values supported?", answer: "Yes. They are gathered through their owning provider before the host callback runs." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "struct",
        target: BuiltinDocumentationLinkTarget::Builtin("struct"),
    },
    BuiltinDocumentationLink {
        label: "fieldnames",
        target: BuiltinDocumentationLinkTarget::Builtin("fieldnames"),
    },
    BuiltinDocumentationLink {
        label: "cellfun",
        target: BuiltinDocumentationLinkTarget::Builtin("cellfun"),
    },
    BuiltinDocumentationLink {
        label: "arrayfun",
        target: BuiltinDocumentationLinkTarget::Builtin("arrayfun"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/structs/core/structfun") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Field traversal, callbacks, multiple outputs, exact collection, errors, and provider behavior", location: "builtins::structs::core::structfun::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed structure, option, callback, output-count, and effect inference", location: "catalog::entries::structs::core::structfun::inference_tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("structfun"), slug: Some("structfun"),
    summary: "Apply a function to each field of a scalar structure.",
    description: "`structfun` invokes a callback for every field and collects each requested output as a column array or a scalar structure.",
    keywords: &["structfun", "structure", "fields", "function handle", "UniformOutput", "ErrorHandler"],
    related: &["struct", "fieldnames", "cellfun", "arrayfun"],
    sections: SECTIONS, examples: examples::EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
