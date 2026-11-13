//! Portable field inventory checks shared by completed semantic-state kinds.

use std::collections::BTreeSet;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues, StateKind};

use super::{AnalysisExecutionError, AnalysisExecutionResult};

pub(in crate::execution) fn validate_portable_fields(
    fields: &[AnalysisField],
) -> AnalysisExecutionResult<()> {
    let mut field_ids = BTreeSet::new();
    for field in fields {
        validate_portable_field(field)?;
        if !field_ids.insert(field.field_id.as_str()) {
            return Err(invalid_field());
        }
    }
    Ok(())
}

/// Checks one field without allocating success-path scratch. Callers validating an inventory
/// must additionally establish uniqueness, either generically or from exact ordinal identities.
pub(in crate::execution) fn validate_portable_field(
    field: &AnalysisField,
) -> AnalysisExecutionResult<()> {
    let AnalysisFieldValues::HostF64(values) = &field.values else {
        return Err(AnalysisExecutionError::Invalid(
            "portable semantic state cannot contain a device-local field reference".into(),
        ));
    };
    let expected = field
        .shape
        .iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(*extent));
    if field.field_id.trim().is_empty()
        || expected != Some(values.len())
        || values.is_empty()
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_field());
    }
    Ok(())
}

fn invalid_field() -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(
        "semantic state fields must be unique, finite, non-empty host arrays with exact shapes"
            .into(),
    )
}

pub(super) fn structural_state_field_ids(
    kind: StateKind,
) -> AnalysisExecutionResult<&'static [&'static str]> {
    match kind {
        StateKind::StructuralPrimary => Ok(&[crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT]),
        _ => Err(AnalysisExecutionError::Invalid(
            "structural linear execution can publish only structural-primary state".into(),
        )),
    }
}

/// One selection authority for publication and its borrowed encoding projection.
pub(super) fn select_structural_state_fields(
    kind: StateKind,
    fields: &[AnalysisField],
) -> AnalysisExecutionResult<Vec<&AnalysisField>> {
    let mut by_id = std::collections::BTreeMap::new();
    for field in fields {
        if by_id.insert(field.field_id.as_str(), field).is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "solver result contains duplicate state field `{}`",
                field.field_id
            )));
        }
    }
    structural_state_field_ids(kind)?
        .iter()
        .map(|field_id| {
            let field = by_id.get(*field_id).copied().ok_or_else(|| {
                AnalysisExecutionError::Invalid(format!(
                    "solver result does not contain state field `{field_id}`"
                ))
            })?;
            validate_portable_field(field)?;
            Ok(field)
        })
        .collect()
}

#[cfg(test)]
mod tests;
