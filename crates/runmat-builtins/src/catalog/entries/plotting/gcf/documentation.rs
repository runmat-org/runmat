use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
BuiltinDocumentationSection { heading: "Behavior",
 paragraphs: &[
"`gcf()` returns a scalar numeric figure handle.",

"The value changes after commands like `figure(n)` or `figure('next')` select a different figure.",

"Subsequent plotting commands target the same current figure until another figure is selected.",

] },

BuiltinDocumentationSection { heading: "Placement and rendering",
 paragraphs: &[
"No numeric provider dispatch is required. Missing figure state is created and its normal host event is emitted.",

"`gcf` returns a host-side numeric handle and does not interact with GPU residency.",

"`gcf` performs no numeric provider dispatch. The returned scalar double handle is host-side graphics identity, not transferable figure state.",

] },

BuiltinDocumentationSection { heading: "Execution and effects",
 paragraphs: &[
"This stateful operation returns existing figure state or creates it and notifies host observers. It does not suspend or run a numeric accelerator kernel.",

"Interpreter and compiled execution use the same typed runtime binding. Zero requested outputs still performs the operation; more than one output rejects before graphics mutation. Numeric handles identify local graphics state, not portable figure values: workers and foreign adapters must not interpret a copied number as transferred graphics ownership.",

"Figure selection and creation reuse the shared registry and host notification path. Repeated calls preserve current axes, figure content, and existing figure identity without a new creation notification.",

"A freshly initialized or fully closed graphics registry receives a valid figure on the next call; use the root CurrentFigure property to inspect without forcing creation.",

] },

];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",

        title: "Read the current figure handle",

        program: "figure(2);\nf = gcf()",

        display_output: Some("f = 2"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::BrowserGraphics,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,

            assertions: "assert(f == 2); assert(strcmp(get(f,'Type'),'figure'));",
        },
    },
    BuiltinExample {
        id: "example-2",

        title: "Observe the handle change after selecting a different figure",

        program: "figure(1);\nf1 = gcf();\nfigure(4);\nf2 = gcf();",

        display_output: Some("f1 = 1\nf2 = 4"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::BrowserGraphics,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,

            assertions: "assert(f1 == 1); assert(f2 == 4); assert(gcf() == f2);",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("gcf"),
 slug: Some("gcf"),
 summary: "Return the current figure handle in MATLAB and RunMat.",
 description: "`gcf` returns the current figure handle, creating missing figure state through the shared figure lifecycle. RunMat represents graphics handles as scalar doubles. Existing figure contents and axes are preserved; figure creation notifies the host but does not itself force pixel presentation.",

keywords: &["gcf",
"current figure",
"figure handle",
"plotting"],
 related: &["figure",
"gca",
"clf"],

sections: SECTIONS,
 examples: EXAMPLES,
 example_exemption: None,

faqs: &[
BuiltinDocumentationFaq { question: "What type of value does `gcf` return?",
 answer: "It returns a scalar double representing the current figure handle." },

BuiltinDocumentationFaq { question: "Does `gcf` create a plot?",
 answer: "It creates a figure when there is no current figure, but does not add a plotted data object. Existing figures are preserved. Pixel presentation belongs to the shared host rendering lifecycle." },

BuiltinDocumentationFaq { question: "Does `gcf` run on the GPU?",
 answer: "No numeric GPU kernel is dispatched. Missing figure state is created on the host and reported to figure observers." },

],

links: &[
BuiltinDocumentationLink { label: "figure",
 target: BuiltinDocumentationLinkTarget::Builtin("figure") },

BuiltinDocumentationLink { label: "gca",
 target: BuiltinDocumentationLinkTarget::Builtin("gca") },

BuiltinDocumentationLink { label: "clf",
 target: BuiltinDocumentationLinkTarget::Builtin("clf") },

BuiltinDocumentationLink { label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/gcf.rs") },

],
 media: &[],

evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/gcf.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantic tests",
 location: "builtins::plotting::gcf::tests" },
 BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
 label: "VM graphics dispatch",
 location: "crates/runmat-vm/tests/plotting.rs" }],
 notes: &["Legacy signatures, behaviors, options, FAQs, examples, links and GPU notes are preserved or corrected against runtime semantics. Typed catalog placement replaces legacy fusion flags. BrowserGraphics examples require actual graphics-host execution; headless runtime tests do not constitute Browser or Desktop presentation evidence. Native Interpreter/JIT/AOT, real-provider gather, and Desktop presentation qualification remain separate required witnesses."],
 },

 introduced: None,
 status: Some(BuiltinDocumentationStatus::Stable),

};
