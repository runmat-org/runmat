use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/minreal.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("minreal"), slug: Some("minreal"),
    summary: "Cancel matching poles and zeros in a transfer function.",
    description: "`minreal(sys)` removes matching pole-zero pairs in an ordinary SISO `tf` model; `minreal(sys,tol)` sets a positive absolute matching tolerance. The default is `sqrt(eps)`.",
    keywords: &["minreal", "pole-zero cancellation", "transfer function", "model reduction"],
    related: &["tf", "pole", "zero"],
    sections: &[
        BuiltinDocumentationSection { heading: "Supported model", paragraphs: &[
            "This implementation supports one SISO transfer-function model with real or complex double coefficients. It preserves the variable, sample time, input delay, and output delay. State-space, zero-pole-gain, model-array, and MIMO reduction remain unsupported.",
            "Cancellation is performed in physical polynomial coordinates, including models whose public coefficients use an inverse discrete variable. For real models, conjugate roots are canceled together so the result remains real.",
        ] },
        BuiltinDocumentationSection { heading: "Resource and numerical boundary", paragraphs: &[
            "The current companion-root calculation has opaque host allocations. Managed finite-memory calls refuse before model-property gathering or output publication; ordinary calls preserve cancellation checks around numerical phases.",
            "Removing a nearby pole and zero changes the transfer function. The tolerance bounds their root distance, not the resulting frequency-response error. Ill-conditioned roots can be rejected instead of publishing a nonfinite or nonreal polynomial.",
            "Malformed model properties and failures in the shared transfer-function parser or companion-root solver retain their `RunMat:tf:*` identifiers. Tolerance, unsupported input class, and pole-zero deflation errors use `RunMat:minreal:*` identifiers.",
        ] },
    ],
    examples: &[BuiltinExample { id: "shared-pole-zero", title: "Cancel a shared pole and zero", program: "sys = tf([1 3 2],[1 4 3]); reduced = minreal(sys);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "[n,d] = tfdata(reduced,'v'); assert(max(abs(n-[1 2])) < 1e-6); assert(max(abs(d-[1 3])) < 1e-6);" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What does tol control?", answer: "It is the maximum absolute distance between a pole and zero eligible for cancellation. The default is sqrt(eps)." }],
    links: &[
        BuiltinDocumentationLink { label: "MathWorks minreal reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/control/ref/dynamicsystem.minreal.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact and near pole-zero cancellation and metadata retention", location: "crates/runmat-runtime/src/builtins/control/minreal.rs" }], notes: &["SISO tf ordinary-memory subset; broader model and managed resource obligations remain open."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
