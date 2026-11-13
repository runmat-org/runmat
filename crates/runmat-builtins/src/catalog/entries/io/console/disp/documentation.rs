use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Value formatting", paragraphs: &[
        "`disp(X)` prints one value to the Command Window without a variable-name header or public output. It accepts exactly one argument; use `fprintf` or `sprintf` for multiple formatted values.",
        "Numeric, logical, and complex arrays use column-major layout, right-aligned columns, and the active numeric `format`. N-D arrays print 2-D pages with `(:,:,[...]) =` headers.",
        "Character arrays print each row without quotes. String scalars and arrays print contents; an empty string prints a blank line and missing strings appear as `<missing>`.",
        "Structures indent fields by four spaces, summarize nested arrays, and render cells as a compact indented grid with MATLAB-style element summaries such as `[2x2 double]`, `'text'`, and `\"string\"`. Nested cells summarize as `[mxn cell]`.",
        "Empty built-in numeric, logical, complex, character, string-array, and cell-array values print no text. Every printed line ends with a newline; use `fprintf` if a trailing newline is unwanted.",
    ] },
    BuiltinDocumentationSection { heading: "Resident values", paragraphs: &[
        "Provider-resident values, including supported nested values, gather through the active acceleration provider before host formatting. This is a sink, not a GPU kernel or fusion operation; no special provider display hook is required.",
        "The provider failure remains an error and is not replayed through another display route. Gathering does not change the source value's ownership or residency.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "message", title: "Print a message", program: "disp(\"Simulation complete.\")", display_output: Some("Simulation complete."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Succeeds },
    BuiltinExample { id: "vector", title: "Display a numeric vector", program: "angles = [0 pi/6 pi/4 pi/3];\ndisp(angles)", display_output: Some("         0    0.5236    0.7854    1.0472"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Succeeds },
    BuiltinExample { id: "matrix", title: "Display a matrix without a variable name", program: "A = [1 23 456; 78 9 10];\ndisp(A)", display_output: Some("     1    23   456\n    78     9    10"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Succeeds },
    BuiltinExample { id: "struct", title: "Print struct fields", program: "cfg = struct(\"solver\", \"bicgstab\", \"tolerance\", 1e-8);\ndisp(cfg)", display_output: Some("    solver: \"bicgstab\"\n  tolerance: 1e-08"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Succeeds },
    BuiltinExample { id: "resident", title: "Display a gpuArray without manual gather", program: "G = gpuArray([1 2; 3 4]);\ndisp(G);             % RunMat gathers and prints host-formatted output automatically", display_output: Some("     1     2\n     3     4"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Succeeds },
    BuiltinExample { id: "strings", title: "Show string and character arrays", program: "names = [\"alpha\" \"beta\"; \"gamma\" \"delta\"];\ndisp(names);\ndisp('RunMat')", display_output: Some("\"alpha\"    \"beta\"\n\"gamma\"    \"delta\"\nRunMat"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Succeeds },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `disp` change the result of my computation?", answer: "No. It is a no-output sink; requesting an output is an error." },
    BuiltinDocumentationFaq { question: "Can I pass multiple arguments to `disp`?", answer: "No. It accepts one value. Use `fprintf` or `sprintf` for multiple formatted values." },
    BuiltinDocumentationFaq { question: "How are structs displayed?", answer: "Fields appear on indented lines. Nested arrays collapse to size/type summaries." },
    BuiltinDocumentationFaq { question: "What happens with empty arrays?", answer: "Empty built-in arrays print no text; an empty string scalar prints a blank line." },
    BuiltinDocumentationFaq { question: "Can `disp` print GPU data directly?", answer: "Yes. It gathers provider-resident values to the host before printing, without changing the source's residency." },
    BuiltinDocumentationFaq { question: "How are missing strings shown?", answer: "They render as `<missing>`." },
    BuiltinDocumentationFaq { question: "Does `disp` honour the current numeric format?", answer: "Yes. `format short`, `format long`, and other active modes affect numeric rendering." },
    BuiltinDocumentationFaq { question: "Does `disp` add a blank line after printing?", answer: "Each printed line ends in a newline. An empty string itself produces a blank line." },
    BuiltinDocumentationFaq { question: "What if the input is a cell array?", answer: "It prints a compact indented view with MATLAB-style element summaries." },
    BuiltinDocumentationFaq { question: "How do I print without a trailing newline?", answer: "Use `fprintf` instead." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "fprintf",
        target: BuiltinDocumentationLinkTarget::Builtin("fprintf"),
    },
    BuiltinDocumentationLink {
        label: "sprintf",
        target: BuiltinDocumentationLinkTarget::Builtin("sprintf"),
    },
    BuiltinDocumentationLink {
        label: "string",
        target: BuiltinDocumentationLinkTarget::Builtin("string"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "input",
        target: BuiltinDocumentationLinkTarget::Builtin("input"),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("disp"), slug: Some("disp"),
    summary: "Display values without returning output in MATLAB and RunMat.",
    description: "`disp` prints a value to the Command Window without its variable name. It formats built-in values on the host and gathers provider-resident values when needed.",
    keywords: &["disp", "display", "print", "console", "gpu"],
    related: &["fprintf", "sprintf", "string", "gather", "input"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Disp binding and formatter", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/disp.rs") },
            BuiltinDocumentationLink { label: "Provider gather dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/dispatcher.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed display and resident formatting", location: "builtins::io::disp::tests" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
