use super::*;
use runmat_types::standard;

#[test]
fn public_variable_identities_are_closed_and_case_insensitive() {
    assert_eq!(
        OptimizableVariableOption::from_public_name(" TRANSFORM "),
        Some(OptimizableVariableOption::Transform)
    );
    assert_eq!(OptimizableVariableOption::from_public_name("Unknown"), None);
    assert_eq!(
        OptimizationVariableType::from_public_name("INTEGER"),
        Some(OptimizationVariableType::Integer)
    );
    assert_eq!(
        OptimizationVariableTransform::from_public_name("LoG"),
        Some(OptimizationVariableTransform::Log)
    );
    assert_eq!(
        standard::OPTIMIZABLE_VARIABLE.display_name(),
        "optimizableVariable"
    );
}
