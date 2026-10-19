use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/dcgain.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dcgain"), slug: Some("dcgain"),
    summary: "Evaluate the steady-state gain of a SISO transfer function.",
    description: "`dcgain(sys)` evaluates a continuous-time SISO `tf` model at `s = 0`, or a discrete-time SISO `tf` model at `z = 1`. The result is a real or complex host scalar. Other LTI model families and MIMO systems are not yet supported.",
    keywords: &["dcgain", "steady state", "control system", "transfer function", "tf"],
    related: &["tf", "stepinfo"],
    sections: &[
        BuiltinDocumentationSection { heading: "Gain and singularities", paragraphs: &[
            "Finite gain comes from direct evaluation of the transfer-function polynomials. An integrator produces an infinite gain; RunMat derives its sign or complex axis from the first nonzero local denominator coefficient at the evaluation point. Exact pole-zero cancellation at that point remains indeterminate. These singularity details are current RunMat behavior, not a claim of verified parity for all MATLAB model forms.",
            "A typed integer array is not a valid system input and does not make `dcgain` an integer-array operation. If a tf object's consumed coefficients or sample time were assigned typed integer values after construction, dcgain checks the corresponding RunMat-only model-property gate and exact conversion at admission.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "The model parser and gain calculation run on the host and do not use a GPU kernel or fuse with elementwise operations. Direct resident numeric inputs are invalid system objects and reject before provider access. Required nested resident coefficient values in a valid tf object are gathered through their owning provider; unrelated properties are not read.",
            "The call may suspend while gathering model coefficients and may throw for invalid or unsupported models or failed gain evaluation. It has no workspace, filesystem, network, callback, user-interface, randomness, or time effect.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "continuous", title: "Continuous-time DC gain", program: "H = tf(2, [1 3]); k = dcgain(H); fprintf('%.4f\\n', k);", display_output: Some("0.6667"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(k - 2/3) < 1e-6);" } },
        BuiltinExample { id: "integrator", title: "Infinite gain of an integrator", program: "H = tf(2, [1 0]); k = dcgain(H); fprintf('%g\\n', k);", display_output: Some("Inf"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isinf(k) && k > 0);" } },
        BuiltinExample { id: "discrete", title: "Discrete-time DC gain", program: "H = tf(1, [1 -0.5], 1); k = dcgain(H); fprintf('%g\\n', k);", display_output: Some("2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(k - 2) < 1e-6);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does dcgain simulate a step response?", answer: "No. It evaluates the transfer-function polynomials at the steady-state point." },
        BuiltinDocumentationFaq { question: "Can dcgain analyze a state-space or MIMO model?", answer: "Not yet. The current implementation accepts scalar SISO tf objects." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "stepinfo", target: BuiltinDocumentationLinkTarget::Builtin("stepinfo") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "SISO gain, integrators and invalid inputs", location: "crates/runmat-runtime/src/builtins/control/dcgain.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Complex singularity and exact cancellation", location: "crates/runmat-runtime/src/builtins/control/tf_model.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM transfer-function workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["The SISO tf-only domain is an explicit partial-coverage limit. Native artifact, browser, and provider behavior require integrated qualification."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
