use super::*;
pub(super) const fn method(title: &'static str, slug: &'static str) -> BuiltinDocumentation {
    BuiltinDocumentation{
        authority:BuiltinDocumentationAuthority::Catalog,
        title:Some(title),slug:Some(slug),
        summary:"Convert or compare a scalar problem-based solver termination value.",
        description:"OptimizationExitflag values identify a supported problem-based solver termination. double returns the documented numeric equivalent, string returns the member name, and eq/ne compare two members of the same canonical enum class.",
        keywords:&["OptimizationExitflag","exitflag","termination","enumeration","double","string","eq","ne"],
        related:&["solve","optimproblem","linprog"],
        sections:&[
            BuiltinDocumentationSection{heading:"Scalar conversion and comparison",paragraphs:&[
                "double(flag) and string(flag) each take one OptimizationExitflag scalar. Equality and inequality take two scalars of that class. The original class registration and invocation context are checked before reading the stored enumeration member.",
                "Equation convergence code 1 is EquationSolved (1), a distinct semantic member from optimization OptimalSolution (1). Equal numeric equivalents do not make the two enum members equal. Equation residual failures retain NoFeasiblePointFound (-2). A numerical scalar bracket convergence is not an exact-zero or global existence certificate.",
                "Problem-based optimization solve emits OptimalSolution (1), SolverLimitExceeded (0), NoFeasiblePointFound (-2), or Unbounded (-3). Numerical evidence that cannot verify a terminal raises a separate error rather than a false FoundNaN or infeasibility member.",
                "Active-set least-squares stops can return IllConditioned (-4) or NoDescentDirectionFound (-8). ObjectiveLimitReached (-3) is a RunMat stop member for an original feasible objective below the selected threshold; it is distinct from Unbounded and carries no unboundedness certificate.",
                "Enum array construction, broad enum reflection, conversion to other numeric classes, and other solver-family exit members are unsupported.",
            ]},
        ],
        examples:&[],example_exemption:Some("Covered by the public solve example and scalar conversion/equality controls."),
        faqs:&[],links:&[BuiltinDocumentationLink{label:"MATLAB problem-based solve outputs",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optim.problemdef.optimizationproblem.solve.html")}],
        media:&[],evidence:BuiltinDocumentationEvidence{
            implementation:&[BuiltinDocumentationLink{label:"Termination value consumer",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimmodel/exitflag.rs")}],
            verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Original model solve and enum controls",location:"builtins::math::optim::optimmodel::lowering::tests"}],
            notes:&[],
        },introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
    }
}
