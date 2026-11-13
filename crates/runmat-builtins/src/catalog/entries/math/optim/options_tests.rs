use super::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver, OptimSwitch};

#[test]
fn option_spellings_decode_once_to_shared_identities() {
    for (text, field) in [
        ("TolX", OptimOptionField::TolX),
        ("stepTolerance", OptimOptionField::TolX),
        ("GradConstr", OptimOptionField::SpecifyConstraintGradient),
        ("FinDiffType", OptimOptionField::FiniteDifferenceType),
        ("FinDiffRelStep", OptimOptionField::FiniteDifferenceStepSize),
        (
            "FiniteDifferenceStepSize",
            OptimOptionField::FiniteDifferenceStepSize,
        ),
        ("TypicalX", OptimOptionField::TypicalX),
        ("FunctionTolerance", OptimOptionField::FunctionTolerance),
        ("OptimalityTolerance", OptimOptionField::OptimalityTolerance),
        ("MAXITER", OptimOptionField::MaxIter),
        ("MaxIterations", OptimOptionField::MaxIter),
        ("OutputFcn", OptimOptionField::OutputFcn),
        ("MaxFunEvals", OptimOptionField::MaxFunEvals),
        ("maxFunctionEvaluations", OptimOptionField::MaxFunEvals),
        ("Display", OptimOptionField::Display),
        ("Algorithm", OptimOptionField::Algorithm),
        ("GradObj", OptimOptionField::SpecifyObjectiveGradient),
        ("HessFcn", OptimOptionField::HessianFcn),
        ("HessMult", OptimOptionField::HessianMultiplyFcn),
        ("HessPattern", OptimOptionField::HessPattern),
        ("MaxPCGIter", OptimOptionField::MaxPCGIter),
        ("TolPCG", OptimOptionField::TolPCG),
        ("PrecondBandWidth", OptimOptionField::PrecondBandWidth),
        ("SubproblemAlgorithm", OptimOptionField::SubproblemAlgorithm),
    ] {
        assert_eq!(OptimOptionField::parse(text), Some(field));
        assert_eq!(OptimOptionField::parse(field.name()), Some(field));
    }
    assert_eq!(OptimOptionField::parse("notAnOption"), None);
    assert_eq!(
        OptimOptionField::parse_for_solver("FunctionTolerance", OptimSolver::Fminunc),
        Some(OptimOptionField::FunctionTolerance)
    );
    assert_eq!(
        OptimOptionField::parse_for_solver("OptimalityTolerance", OptimSolver::Fminunc),
        Some(OptimOptionField::OptimalityTolerance)
    );
    for solver in [
        OptimSolver::Coneprog,
        OptimSolver::Fsolve,
        OptimSolver::Lsqcurvefit,
    ] {
        assert_eq!(
            OptimOptionField::parse_for_solver("FunctionTolerance", solver),
            Some(OptimOptionField::TolFun)
        );
    }
    assert_eq!(
        OptimOptionField::parse_for_solver("FunctionTolerance", OptimSolver::Lsqnonlin),
        Some(OptimOptionField::FunctionTolerance)
    );
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
    assert!(OptimSolver::Fminunc.accepts_algorithm(OptimAlgorithm::TrustRegion));
    assert!(!OptimSolver::Lsqnonlin.accepts_algorithm(OptimAlgorithm::TrustRegion));
    assert!(super::OptimHessianSource::parse("objective").is_some());
    assert_eq!(
        super::OptimSubproblemAlgorithm::parse("factorization"),
        Some(super::OptimSubproblemAlgorithm::Factorization)
    );
    for field in OptimOptionField::ALL {
        assert_eq!(OptimOptionField::parse(field.name()), Some(field));
    }
    assert!(OptimSolver::Fminunc.accepts(OptimOptionField::OutputFcn));
    assert!(OptimSolver::Fsolve.accepts(OptimOptionField::SpecifyObjectiveGradient));
    assert!(OptimSolver::Linprog.accepts(OptimOptionField::MaxIter));
    assert!(!OptimSolver::Linprog.accepts(OptimOptionField::OutputFcn));
    assert!(OptimSolver::Linprog.accepts_zero_iterations());
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

#[test]
fn diagnose_display_is_named_but_not_admitted_by_legacy_profiles() {
    assert_eq!(
        OptimDisplay::parse(" DIAGNOSE "),
        Some(OptimDisplay::Diagnose)
    );
    assert_eq!(OptimDisplay::Diagnose.name(), "diagnose");
    for solver in [
        OptimSolver::Generic,
        OptimSolver::Fminbnd,
        OptimSolver::Fzero,
        OptimSolver::Fminunc,
        OptimSolver::Fseminf,
        OptimSolver::Fsolve,
        OptimSolver::Quadprog,
        OptimSolver::Particleswarm,
    ] {
        assert!(!solver.accepts_display(OptimDisplay::Diagnose));
        assert!(solver.accepts_display(OptimDisplay::Final));
    }
}
