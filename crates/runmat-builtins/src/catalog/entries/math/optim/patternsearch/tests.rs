use crate::*;
#[test]
fn classic_profile_has_one_default_range_and_spelling_authority() {
    for field in OptimOptionField::ALL
        .into_iter()
        .filter(|&field| OptimSolver::Patternsearch.accepts(field))
    {
        assert!(
            OptimPatternDefault::for_field(field).is_some(),
            "{}",
            field.name()
        );
        assert_eq!(
            OptimOptionField::parse_for_solver(
                field.name_for_solver(OptimSolver::Patternsearch),
                OptimSolver::Patternsearch
            ),
            Some(field)
        );
    }
    assert!(OptimSolver::Patternsearch.accepts_display(OptimDisplay::Diagnose));
    assert!(!OptimSolver::Particleswarm.accepts_display(OptimDisplay::Diagnose));
    assert_eq!(
        OptimOptionField::parse_for_solver("TolFun", OptimSolver::Patternsearch),
        Some(OptimOptionField::FunctionTolerance)
    );
    assert_eq!(
        OptimOptionField::parse_for_solver("OptimalityTolerance", OptimSolver::Patternsearch),
        None
    );
    assert!(
        OptimPatternScalarRule::for_field(OptimOptionField::MaxMeshSize)
            .unwrap()
            .accepts(f64::INFINITY)
    );
    assert!(
        !OptimPatternScalarRule::for_field(OptimOptionField::MaxMeshSize)
            .unwrap()
            .accepts(0.0)
    );
    assert!(OptimPatternScalarRule::for_field(OptimOptionField::MaxTime)
        .unwrap()
        .accepts(f64::INFINITY));
    assert!(!OptimPatternScalarRule::for_field(OptimOptionField::TolCon)
        .unwrap()
        .accepts(f64::INFINITY));
}
