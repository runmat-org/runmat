use crate::*;
mod algorithm;
mod examples;
pub(super) const SOLVER: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ga"), slug: Some("ga"),
    summary: "Minimize scalar fitness by genetic selection, crossover and mutation with continuous or integer constraints.",
    description: "Evolve a retained population using measured or explicitly supplied fitness, original linear/integer domains and nonlinear feasibility. Return the best feasible observation and the last completed population.",
    keywords: &["genetic algorithm","mixed integer","nonlinear constraints","population","optimization"],
    related: &["ga","optimoptions","particleswarm","patternsearch","intlinprog","fmincon","rng"],
    sections: algorithm::SECTIONS, examples: examples::EXAMPLES, example_exemption: None,
    faqs: &[], media: &[],
    links: &[
        BuiltinDocumentationLink { label: "MathWorks ga calling forms", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/ga.html") },
        BuiltinDocumentationLink { label: "MathWorks genetic options and custom callbacks", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/genetic-algorithm-options.html") },
        BuiltinDocumentationLink { label: "MathWorks nonlinear genetic algorithms", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/description-of-the-nonlinear-constraint-solver.html") },
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Genetic solver and modular controller", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/ga.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Initial score provenance and completed normal/nonlinear lifecycle", location: "builtins::math::optim::ga::controller::driver::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original linear rows, zero nonlinear tolerance and exact feasible directions", location: "builtins::math::optim::ga::linear::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Genetic numerical operators and nonlinear merit", location: "builtins::math::optim::ga::operators" },
        ], notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
