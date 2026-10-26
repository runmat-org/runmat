use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("secondordercone"), slug: Some("secondordercone"),
 summary: "Create a second-order cone constraint descriptor for coneprog.",
 description: "`secondordercone(A,b,d,gamma)` creates a constraint descriptor for `norm(A*x - b) <= d'*x - gamma`. Pass the result, or a cell array of such results, to `coneprog`.",
 keywords: &["secondordercone", "coneprog", "second-order cone", "SOCP", "optimization"],
 related: &["coneprog"],
 sections: &[BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A must be a nonempty real dense or sparse numeric matrix. The descriptor preserves dense versus sparse storage while normalizing numeric coefficients to double.",
            "`b` must be a numeric vector with length equal to `size(A,1)`; `d` must be a numeric vector with length equal to `size(A,2)`; `gamma` must be a finite numeric scalar.",
            "The returned descriptor is a host struct with A, b, d, gamma and Type fields. Pass it, an array of descriptors or a cell array directly to coneprog. The center b is subtracted from A*x and gamma is subtracted from d'*x; gamma=-1 with zero d describes a unit-radius cone section.",
            "Supported native integer coefficients are read from native storage and normalized to binary64 with its ordinary rounding. Wide integer coefficients need not remain exact; the host numerical output is double. This preserves RunMat's floating-boundary behavior and does not assert MATLAB integer-overload parity."
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution",
        paragraphs: &[
            "The host numerical boundary gathers resident numeric operands. No device solver, elementwise fusion, reduction fusion, broadcasting solver, or distributed solver is provided.",
            "The constructor requires exactly four arguments A, b, d, gamma. Finite host-memory execution is unsupported and refused before gather or descriptor allocation; no complete descriptor backing bound is claimed. Numeric constants remain host data; no device constant buffer or shader ABI is promised."
        ],
    }],
 examples: &[
BuiltinExample { id: "example-1", title: "Create a unit-disk cone", program: "soc = secondordercone(eye(2), [0; 0], [0; 0], -1)", display_output: Some("soc =\n  struct with fields:\n      A: [2x2 double]\n      b: [2x1 double]\n      d: [2x1 double]\n  gamma: -1\n   Type: \"secondordercone\""), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(soc.A,eye(2))); assert(isequal(soc.b,[0;0])); assert(isequal(soc.d,[0;0])); assert(soc.gamma==-1); assert(strcmp(soc.Type,'secondordercone'));" } },
 ],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Can I pass multiple cones to coneprog?", answer: "Yes. Pass a descriptor array or cell array containing multiple secondordercone descriptors." }],
 links: &[BuiltinDocumentationLink { label: "coneprog", target: BuiltinDocumentationLinkTarget::Builtin("coneprog") },BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/coneprog.rs") }],
 media: &[],
 evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/coneprog.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime behavior", location: "builtins::math::optim::coneprog::tests" }], notes: &[] },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
