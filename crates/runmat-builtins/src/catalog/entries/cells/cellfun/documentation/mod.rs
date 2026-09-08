mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Cell-wise callback execution",
        paragraphs: &[
            "`Y = cellfun(func,C1,C2,...)` calls `func` once per cell position, passing the contents of each input cell. All cell-array inputs must have the same shape. Once a non-cell argument appears, it and the remaining non-cell arguments are constants passed to every invocation.",
            "Function handles, anonymous and captured functions, and ordinary function names use RunMat's normal callable-resolution rules. The documented shorthand names `isempty`, `islogical`, `isreal`, `length`, `ndims`, `prodofsize`, `size`, and `isclass` are accepted without an `@` prefix.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Output and error handling",
        paragraphs: &[
            "`UniformOutput` defaults to true. Each callback must return one scalar compatible with the collected result class. Set `'UniformOutput', false` when callbacks return arrays, strings, structures, resident values, or heterogeneous values; the outer result is then a cell array with the input shape.",
            "`ErrorHandler` receives a structure with `identifier`, `message`, `index`, and `indices`, followed by the cell contents and constant arguments for the failed position. Its return value is collected in place of the failed callback result.",
            "Fixed-width integer results retain one exact common class. Logical and double results may promote according to the uniform collection contract, and real results may promote to a matching complex representation. Multiple callback outputs remain unsupported; the current runtime requests one result per cell.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Provider-resident values",
        paragraphs: &[
            "`cellfun` is host-controlled. Provider-resident values stored inside input cells or passed as constants are gathered through their owning provider before callback invocation. Uniform packed output is host-resident.",
            "With `'UniformOutput', false`, callback results are inserted into the result cell without an automatic gather, so a returned provider-resident value keeps its residency. Callback execution makes `cellfun` a fusion boundary.",
        ],
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does cellfun modify its input cells?", answer: "No. It reads each cell value and creates a new result." },
    BuiltinDocumentationFaq { question: "Can callbacks return strings or structures?", answer: "Yes with `'UniformOutput', false`, which retains each result in a cell." },
    BuiltinDocumentationFaq { question: "Do constant arguments need the cell-array shape?", answer: "No. Non-cell arguments following the cell inputs are passed unchanged to every callback invocation." },
    BuiltinDocumentationFaq { question: "What happens for empty inputs?", answer: "The result keeps the empty input shape. Nonuniform mode returns an empty cell array, and uniform mode returns an empty double array." },
    BuiltinDocumentationFaq { question: "How does cellfun differ from arrayfun?", answer: "cellfun passes each cell's stored value to the callback. arrayfun extracts scalar elements from numeric, logical, character, string, or complex arrays." },
    BuiltinDocumentationFaq { question: "Can callbacks capture variables?", answer: "Yes. Anonymous functions and other closures retain their captured values and receive each cell's contents after those captures." },
    BuiltinDocumentationFaq { question: "Which names can omit the function-handle prefix?", answer: "The documented shorthand set is `isempty`, `islogical`, `isreal`, `length`, `ndims`, `prodofsize`, `size`, and `isclass`. Ordinary named callbacks also follow the session's callable-resolution rules." },
    BuiltinDocumentationFaq { question: "When is a loop clearer?", answer: "Use a loop when the body has several steps, mutates surrounding state, or needs custom control flow. Use cellfun for a compact cell-wise callback with a regular collection contract." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "cell",
        target: BuiltinDocumentationLinkTarget::Builtin("cell"),
    },
    BuiltinDocumentationLink {
        label: "cell2mat",
        target: BuiltinDocumentationLinkTarget::Builtin("cell2mat"),
    },
    BuiltinDocumentationLink {
        label: "mat2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("mat2cell"),
    },
    BuiltinDocumentationLink {
        label: "arrayfun",
        target: BuiltinDocumentationLinkTarget::Builtin("arrayfun"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/cells/core/cellfun") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Cell traversal, callbacks, output collection, error handling, and provider behavior", location: "builtins::cells::core::cellfun::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed cell, option, callback, shape, and effect inference", location: "catalog::entries::cells::cellfun::inference_tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cellfun"),
    slug: Some("cellfun"),
    summary: "Apply a function to values stored in equal-sized cell arrays.",
    description: "`cellfun` invokes a callback for each cell position and collects either a uniform array or a cell array of individual results.",
    keywords: &["cellfun", "cell arrays", "function handle", "UniformOutput", "ErrorHandler"],
    related: &["cell", "cell2mat", "mat2cell", "arrayfun", "gpuArray", "gather"],
    sections: SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
