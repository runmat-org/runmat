use super::*;
use runmat_types::standard;

#[test]
fn public_bayesian_option_identity_is_closed() {
    assert_eq!(
        BayesoptOption::from_public_name("MAXOBJECTIVEEVALUATIONS"),
        Some(BayesoptOption::MaxObjectiveEvaluations)
    );
    assert_eq!(BayesoptOption::from_public_name("unknown"), None);
    assert_eq!(
        BayesoptAcquisition::from_public_name("expected_improvement_plus"),
        Some(BayesoptAcquisition::ExpectedImprovementPlus)
    );
    assert_eq!(
        standard::BAYESIAN_OPTIMIZATION.display_name(),
        "BayesianOptimization"
    );
}
