use runmat_analysis_core::{AnalysisModel, PrescribedValue, StructuralConstraintKind};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub(super) enum TransientConstraintAdmissionError {
    #[error(
        "transient structural constraint `{constraint_id}` prescribes time-varying {quantity}; transient derivatives and base-motion realization are not implemented"
    )]
    TimeVaryingPrescribedKinematics {
        constraint_id: String,
        quantity: &'static str,
    },
}

pub(super) fn validate(model: &AnalysisModel) -> Result<(), TransientConstraintAdmissionError> {
    for constraint in model
        .constraints
        .iter()
        .filter_map(|value| value.structural())
    {
        match &constraint.kind {
            StructuralConstraintKind::Prescribed(values) => {
                for translation in &values.translations {
                    require_study_constant(
                        &translation.value,
                        model,
                        &constraint.constraint_id,
                        "displacement",
                    )?;
                }
                for rotation in &values.rotations {
                    require_study_constant(
                        &rotation.value,
                        model,
                        &constraint.constraint_id,
                        "rotation",
                    )?;
                }
            }
            StructuralConstraintKind::NormalDisplacement { value, .. } => {
                require_study_constant(value, model, &constraint.constraint_id, "displacement")?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn require_study_constant<T: runmat_analysis_core::PrescribedQuantity>(
    value: &PrescribedValue<T>,
    model: &AnalysisModel,
    constraint_id: &str,
    quantity: &'static str,
) -> Result<(), TransientConstraintAdmissionError> {
    if value.study_constant(&model.signals).is_some() {
        Ok(())
    } else {
        Err(
            TransientConstraintAdmissionError::TimeVaryingPrescribedKinematics {
                constraint_id: constraint_id.into(),
                quantity,
            },
        )
    }
}

#[cfg(test)]
mod tests;
