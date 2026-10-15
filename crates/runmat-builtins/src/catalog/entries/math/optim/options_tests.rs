use super::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver, OptimSwitch};

#[test]
fn option_spellings_decode_once_to_shared_identities() {
    for (text, field) in [
        ("TolX", OptimOptionField::TolX),
        ("stepTolerance", OptimOptionField::TolX),
        ("FunctionTolerance", OptimOptionField::TolFun),
        ("OptimalityTolerance", OptimOptionField::TolFun),
        ("MAXITER", OptimOptionField::MaxIter),
        ("MaxFunEvals", OptimOptionField::MaxFunEvals),
        ("Display", OptimOptionField::Display),
        ("Algorithm", OptimOptionField::Algorithm),
        ("GradObj", OptimOptionField::SpecifyObjectiveGradient),
    ] {
        assert_eq!(OptimOptionField::parse(text), Some(field));
        assert_eq!(OptimOptionField::parse(field.name()), Some(field));
    }
    assert_eq!(OptimOptionField::parse("notAnOption"), None);
}

#[test]
fn solver_policies_keep_typed_option_and_control_identities() {
    assert_eq!(OptimSolver::parse(" FMINBND "), Some(OptimSolver::Fminbnd));
    assert_eq!(OptimSolver::parse("notASolver"), None);
    assert!(OptimSolver::Fminbnd.accepts(OptimOptionField::MaxIter));
    assert!(!OptimSolver::Fminbnd.accepts(OptimOptionField::Algorithm));
    assert!(OptimSolver::Fminbnd.accepts_display(OptimDisplay::Notify));
    assert!(!OptimSolver::Fzero.accepts_display(OptimDisplay::Notify));
    assert!(OptimSolver::Fminunc.accepts_algorithm(OptimAlgorithm::Bfgs));
    assert!(!OptimSolver::Fzero.accepts_algorithm(OptimAlgorithm::Bfgs));
    for solver in [OptimSolver::Lsqcurvefit, OptimSolver::Lsqnonlin] {
        assert!(solver.accepts_algorithm(OptimAlgorithm::LevenbergMarquardt));
        assert!(solver.accepts_algorithm(OptimAlgorithm::TrustRegionReflective));
        assert!(!solver.accepts_algorithm(OptimAlgorithm::Bfgs));
        assert!(!solver.accepts_algorithm(OptimAlgorithm::QuasiNewton));
    }
    assert!(OptimSolver::Generic.accepts_algorithm(OptimAlgorithm::Bfgs));
    assert_eq!(OptimDisplay::parse(" iter "), Some(OptimDisplay::Iter));
    assert_eq!(OptimAlgorithm::parse(" bfgs "), Some(OptimAlgorithm::Bfgs));
    assert_eq!(OptimSwitch::parse("yes"), Some(OptimSwitch::On));
    assert!(!OptimSwitch::Off.enabled());
}
