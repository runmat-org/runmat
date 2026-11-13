//! Structural sparse-shape validation shared by operator demand planning.

use super::{mul, ResourcePlanningError, SparseResourceFacts};

pub(super) fn validate_sparse(facts: &SparseResourceFacts) -> Result<(), ResourcePlanningError> {
    if facts.rows.0 == 0
        || facts.columns.0 == 0
        || facts.operator_nonzeros.is_empty()
        || facts.operator_nonzeros.iter().any(|count| count.0 == 0)
    {
        return Err(ResourcePlanningError::Invalid(
            "sparse dimensions and every operator nonzero count must be nonzero".into(),
        ));
    }
    let dense = mul(facts.rows.0, facts.columns.0, "sparse dense bound")?;
    if facts.operator_nonzeros.iter().any(|count| count.0 > dense) {
        return Err(ResourcePlanningError::Invalid(
            "sparse nonzeros exceed matrix dimensions".into(),
        ));
    }
    Ok(())
}
