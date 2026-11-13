use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/pzmap.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pzmap"), slug: Some("pzmap"),
    summary: "Plot or return poles and zeros of a SISO dynamic model.",
    description: "`pzmap(sys)` plots pole and zero locations for a supported SISO `tf` or `ss` model. With one output it returns poles; with two, `[p,z]` returns separate pole and zero columns. Statement form accepts multiple models with optional line styles.",
    keywords: &["pzmap", "pole zero map", "poles", "zeros", "control system", "transfer function", "state space", "tf", "ss"],
    related: &["tf", "ss", "pole", "zero", "plot"],
    sections: &[
        BuiltinDocumentationSection { heading: "Model and output forms", paragraphs: &[
            "A scalar SISO tf model can have real or complex coefficients. A scalar SISO ss model is converted to a transfer function before root extraction; MIMO and model-array forms remain unsupported. Nonzero input or output delay is unsupported. The returned columns are real double when their roots are real within the root tolerance, complex double otherwise, and empty 0-by-1 for a static gain.",
            "One and two requested outputs are data-returning forms. Exactly zero outputs draws poles as x markers and zeros as o markers. Plot form accepts `pzmap(sys1,'r',sys2,'b')`; each system can have one style argument. More than two requested outputs and extra arguments in data-returning forms raise an error.",
            "An integer array is not a dynamic-system object. Typed-integer tf coefficients or Ts and consumed ss matrices or Ts assigned after construction are checked when pzmap admits those model fields, including exact conversion to the host model's numeric domain. Logical tf coefficients have a separate model-property gate.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Root computation runs on the host; no GPU kernel or fusion applies. Required tf or ss properties gather through their owner after class and property preflight. Unrelated object properties are not downloaded. Returned tensors are host-owned and do not alias the model.",
            "Data-returning forms are pure apart from possible suspension and errors. Zero-output forms mutate graphics state and require a UI-capable host; they may suppress known nonfatal presentation setup errors. Plot state and rendering stay with the existing graphics service.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "data", title: "Return poles and zeros", program: "H = tf([1 3 2], [1 4]); [p,z] = pzmap(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [1 1])); assert(abs(p(1)+4) < 1e-8); assert(isequal(size(z), [2 1])); assert(max(abs(sort(z)-[-2;-1])) < 1e-8);" } },
        BuiltinExample { id: "state-space", title: "State-space pole and zero columns", program: "sys = ss(-1, 1, 1, 0); [p,z] = pzmap(sys);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [1 1])); assert(abs(p(1)+1) < 1e-8); assert(isequal(size(z), [0 1]));" } },
        BuiltinExample { id: "complex-poles", title: "Complex pole locations", program: "H = tf(1, [1 0 1]); [p,z] = pzmap(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [2 1])); assert(max(abs(sort(imag(p))-[-1;1])) < 1e-8); assert(isequal(size(z), [0 1]));" } },
        BuiltinExample { id: "plot", title: "Plot a pole-zero map", program: "H = tf([1 2], [1 3 4]); pzmap(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(pole(H)), [2 1])); assert(isequal(size(zero(H)), [1 1]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does pzmap support state-space models?", answer: "Yes, for SISO ss models. RunMat converts their state-space matrices to a transfer function for zero extraction; MIMO models remain unsupported." },
        BuiltinDocumentationFaq { question: "What does pzmap return with one output?", answer: "It returns the pole column. Request [p,z] = pzmap(sys) for zeros as well." },
        BuiltinDocumentationFaq { question: "Does pzmap run on the GPU?", answer: "No. Model data may gather from an owning provider, then root calculation and plotting run on the host." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") },
        BuiltinDocumentationLink { label: "pole", target: BuiltinDocumentationLinkTarget::Builtin("pole") },
        BuiltinDocumentationLink { label: "zero", target: BuiltinDocumentationLinkTarget::Builtin("zero") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "tf/ss roots, output arity, plotting and resident-object preflight", location: "crates/runmat-runtime/src/builtins/control/pzmap.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["SISO/model-array coverage is partial; UI rendering, native artifacts, browser and real-provider behavior need integrated qualification."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
