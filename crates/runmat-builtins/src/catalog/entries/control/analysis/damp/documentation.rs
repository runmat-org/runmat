use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/damp.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("damp"),
    slug: Some("damp"),
    summary: "Return natural frequencies, damping ratios, and poles for control models.",
    description: "`damp(sys)` derives natural frequencies and damping ratios from the poles of a local `tf` or `ss` model. It returns up to three host-resident column vectors: natural frequency `wn`, damping ratio `zeta`, and poles `p`.",
    keywords: &["damp", "damping ratio", "natural frequency", "poles", "control system", "tf", "ss"],
    related: &["pole", "tf", "ss"],
    sections: &[
        BuiltinDocumentationSection { heading: "Models and outputs", paragraphs: &[
            "The supported inputs are SISO tf objects and ss state-space objects. For transfer functions, damp analyzes the stored denominator polynomial without canceling pole-zero pairs. For state-space models, it analyzes the state matrix eigenvalues. Static-gain transfer functions produce empty N-by-1 columns with N equal to zero.",
            "wn and zeta are real double N-by-1 columns. Pole output p is a real double column when all poles are real, or a complex double column when any pole has a nonzero imaginary component. One, two, or three requested outputs return the corresponding prefix; requesting more than three raises an output-count error. A discarded-output call still analyzes the model, while a statement call does not yet render a diagnostic table.",
        ] },
        BuiltinDocumentationSection { heading: "Continuous and discrete damping", paragraphs: &[
            "For a continuous-time pole p, wn = abs(p) and zeta = -real(p)/wn. For a discrete-time model with sample time Ts, the damping metrics use the equivalent continuous pole log(p)/Ts; the returned p remains in the discrete domain. Modes are sorted by natural frequency, then damping ratio and pole components.",
            "Special poles use defined current behavior: a continuous zero or infinity has damping ratio -1; a discrete zero has infinite natural frequency and damping ratio +1; a discrete one or infinity has damping ratio -1. At a continuous zero and discrete one, natural frequency is zero. Numerical behavior for singular and nearly singular poles remains part of the current control-model implementation.",
        ] },
        BuiltinDocumentationSection { heading: "Placement and current limits", paragraphs: &[
            "damp reads host-side model metadata and runs pole analysis on the CPU. It does not execute a GPU kernel, join fusion, or download a direct numeric gpuArray input: raw integer, logical, complex, and resident numeric arrays are not model objects and fail before provider access. Numeric-class support belongs to tf and ss construction, before damp receives the model.",
            "This call does not suspend, invoke callbacks, access workspace or environment state, or require filesystem, network, time, randomness, or a user interface. It can throw for malformed models or failed pole extraction. Distributed model objects and transfer to remote workers are not supported.",
            "Generalized, uncertain, zpk, sparss, and mechss models are not yet supported. Sparse-model Focus, MaxNumber, Shift, Tolerance, and Display options are not implemented. Statement-form damp(sys) does not yet show MATLAB's diagnostic table; request one to three outputs to use the result.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "second-order", title: "Damping of a second-order transfer function", program: "G = tf(1, [100 50 400]); [wn, zeta, p] = damp(G);", display_output: Some("Both natural frequencies are near 2, with damping ratios near 0.125."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(wn), [2 1])); assert(isequal(size(zeta), [2 1])); assert(isequal(size(p), [2 1])); assert(all(abs(wn-2)<1e-8)); assert(all(abs(zeta-0.125)<1e-8)); assert(~isreal(p));" } },
        BuiltinExample { id: "discrete", title: "Discrete-time damping and original pole", program: "D = tf(1, [1 -0.5], 0.1); [wn, zeta, p] = damp(D);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(wn(1)-6.931471805599453)<1e-10); assert(abs(zeta(1)-1)<1e-12); assert(abs(p(1)-0.5)<1e-12);" } },
        BuiltinExample { id: "state-space", title: "Analyze state-space modes", program: "S = ss([0 1; -4 -0.5], [0; 1], [1 0], 0); [wn, zeta] = damp(S);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(wn), [2 1])); assert(all(abs(wn-2)<1e-8)); assert(all(abs(zeta-0.125)<1e-8));" } },
        BuiltinExample { id: "static-gain", title: "Static gain has no dynamic modes", program: "[wn, zeta, p] = damp(tf(2, 1));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(wn)); assert(isempty(zeta)); assert(isempty(p)); assert(isequal(size(wn), [0 1]));" } },
        BuiltinExample { id: "invalid-numeric", title: "A numeric array is not a control model", program: "damp(1)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:damp:InvalidModel" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does damp cancel pole-zero pairs?", answer: "No. For tf objects it analyzes the stored denominator polynomial." },
        BuiltinDocumentationFaq { question: "Does damp support discrete-time systems?", answer: "Yes. Frequency and damping ratio use log(p)/Ts while the returned pole remains in the discrete domain." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pole", target: BuiltinDocumentationLinkTarget::Builtin("pole") },
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Transfer-function and state-space poles, output counts, special poles, and invalid inputs", location: "crates/runmat-runtime/src/builtins/control/damp.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM transfer-function workflow", location: "crates/runmat-vm/tests/control.rs" },
        ],
        notes: &["tf/ss-only model support, unimplemented sparse options, and statement display remain explicit partial limits. Portable examples and native artifact paths require integrated execution."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
