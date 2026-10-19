use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/rlocus.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rlocus"), slug: Some("rlocus"),
    summary: "Compute or plot the root locus of a SISO dynamic model.",
    description: "`rlocus(sys[,k])` computes roots of `den(sys)+k*num(sys)` for each nonnegative gain. It accepts supported scalar SISO `tf` and `ss` models, returning a branches-by-gains root matrix and optionally the gain row. With no requested outputs it plots the trajectories and open-loop poles and zeros.",
    keywords: &["rlocus", "root locus", "control system", "transfer function", "tf", "ss"],
    related: &["tf", "ss", "feedback", "plot"],
    sections: &[
        BuiltinDocumentationSection { heading: "Model and gain forms", paragraphs: &[
            "An explicit k must be a nonempty finite nonnegative numeric vector. With k omitted, RunMat chooses a deterministic logarithmic gain grid and samples around real positive breakaway candidates. A typed-integer gain vector is a RunMat-only extension: every value must be exactly representable as double, and strict MATLAB mode rejects it before model or provider access. Separately, typed-integer tf coefficients or Ts, logical tf coefficients, and typed-integer consumed ss matrices or Ts assigned after construction use their model-property gates; integer model fields also require exact conversion.",
            "tf models may have real or complex coefficients, and may be continuous or discrete. SISO ss models are converted to equivalent numerator and denominator polynomials. MIMO, model arrays, and nonzero input or output delays remain unsupported. Root columns are real double when all branches are real within the root tolerance, complex double otherwise; k is a 1-by-N double row.",
            "One or two requested outputs return roots and optionally gains. The current runtime pads requests beyond two outputs with numeric zero. No-output statement form accepts multiple models, optional styles, and one shared gain vector following all models.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Model fields and an explicit resident gain vector gather through their owning provider after admission; unrelated object properties are not read. Polynomial conversion, gain-grid construction, branch tracking and root solves run on CPU, with no GPU kernel or fusion. Outputs are new host-owned tensors.",
            "No-output calls change graphics state and require a UI-capable host; data-returning forms do not. Both forms can suspend while gathering data and can throw for malformed models, gain vectors or failed root calculations. The shared plotting service owns color and hold state.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "explicit-gains", title: "Roots at selected gains", program: "H = tf(1, [1 1]); [r,k] = rlocus(H, [0 1 3]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), [1 3])); assert(max(abs(r-[-1 -2 -4])) < 1e-8); assert(isequal(k, [0 1 3]));" } },
        BuiltinExample { id: "discrete", title: "Discrete-time transfer function", program: "H = tf(1, [1 -0.5], 0.1); r = rlocus(H, 0.5);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), [1 1])); assert(abs(r(1)) < 1e-8);" } },
        BuiltinExample { id: "integer-gain", title: "RunMat integer gain extension", program: "H = tf(1, [1 1]); [r,k] = rlocus(H, uint16([0 1 3]));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), [1 3])); assert(isequal(k, [0 1 3])); assert(strcmp(class(k), 'double'));" } },
        BuiltinExample { id: "plot", title: "Plot a root locus", program: "s = tf('s'); H = (s + 2) / (s^2 + 3*s + 4); rlocus(H);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(pole(H)), [2 1])); [r,k] = rlocus(H, [0 1 3]); assert(isequal(size(r), [2 3])); assert(max(abs(sum(r,1) + 3 + k)) < 1e-8); assert(max(abs(prod(r,1) - 4 - 2*k)) < 1e-8);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What equation does rlocus solve?", answer: "For each gain k, it finds the roots of the negative-unity-feedback characteristic polynomial den(sys)+k*num(sys)." },
        BuiltinDocumentationFaq { question: "Does rlocus support state-space or MIMO models?", answer: "It supports SISO ss models by converting their state-space matrices to control polynomials. MIMO models and model arrays remain unsupported." },
        BuiltinDocumentationFaq { question: "Does it run on the GPU?", answer: "No. Resident inputs can gather, then model conversion, roots and plotting run on the host." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") },
        BuiltinDocumentationLink { label: "feedback", target: BuiltinDocumentationLinkTarget::Builtin("feedback") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "gain admission, tf/ss roots, branches, plotting, and resident preflight", location: "crates/runmat-runtime/src/builtins/control/rlocus.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["SISO/model-array coverage and current plotting/output-padding behavior leave this partial. Integrated browser, provider and native-artifact gates remain required."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
