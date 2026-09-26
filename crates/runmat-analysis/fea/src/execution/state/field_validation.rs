//! Portable field inventory checks shared by completed semantic-state kinds.

use std::collections::BTreeSet;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues, StateKind};

use super::{AnalysisExecutionError, AnalysisExecutionResult};

pub(in crate::execution) fn validate_portable_fields(
    fields: &[AnalysisField],
) -> AnalysisExecutionResult<()> {
    let mut field_ids = BTreeSet::new();
    for field in fields {
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
            || !field_ids.insert(field.field_id.as_str())
            || expected != Some(values.len())
            || values.is_empty()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(AnalysisExecutionError::Invalid(
                "semantic state fields must be unique, finite, non-empty host arrays with exact shapes"
                    .into(),
            ));
        }
    }
    Ok(())
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
