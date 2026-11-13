use runmat_analysis_core::SpectrumResponse;
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::FeaRunError,
    pipeline::structural_nodal::{nodal_vector_shape, project_structural_nodal_snapshot},
};

use super::super::ModalResponseSpectrumResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumNodalQuantity {
    RelativeDisplacement,
    Rotation,
    RelativeVelocity,
    RelativeAngularVelocity,
    RelativeAcceleration,
    RelativeAngularAcceleration,
    AbsoluteAcceleration,
    AbsoluteAngularAcceleration,
    PseudoVelocity,
    PseudoAngularVelocity,
    PseudoAcceleration,
    PseudoAngularAcceleration,
}

impl ResponseSpectrumNodalQuantity {
    pub fn layout(self) -> runmat_analysis_core::AnalysisFieldLayout {
        use runmat_analysis_core::ObservationQuantity as Q;
        let quantity = match self {
            Self::RelativeDisplacement => Q::Displacement,
            Self::Rotation => Q::Rotation,
            Self::RelativeVelocity | Self::PseudoVelocity => Q::Velocity,
            Self::RelativeAngularVelocity | Self::PseudoAngularVelocity => Q::AngularVelocity,
            Self::RelativeAcceleration | Self::AbsoluteAcceleration | Self::PseudoAcceleration => {
                Q::Acceleration
            }
            Self::RelativeAngularAcceleration
            | Self::AbsoluteAngularAcceleration
            | Self::PseudoAngularAcceleration => Q::AngularAcceleration,
        };
        runmat_analysis_core::AnalysisFieldLayout::vector3(
            quantity,
            runmat_analysis_core::FieldLocation::Node,
        )
    }

    pub const fn for_translation(response: SpectrumResponse) -> Self {
        match response {
            SpectrumResponse::RelativeDisplacement => Self::RelativeDisplacement,
            SpectrumResponse::RelativeVelocity => Self::RelativeVelocity,
            SpectrumResponse::RelativeAcceleration => Self::RelativeAcceleration,
            SpectrumResponse::AbsoluteAcceleration => Self::AbsoluteAcceleration,
            SpectrumResponse::PseudoVelocity => Self::PseudoVelocity,
            SpectrumResponse::PseudoAcceleration => Self::PseudoAcceleration,
        }
    }

    pub const fn for_rotation(response: SpectrumResponse) -> Self {
        match response {
            SpectrumResponse::RelativeDisplacement => Self::Rotation,
            SpectrumResponse::RelativeVelocity => Self::RelativeAngularVelocity,
            SpectrumResponse::RelativeAcceleration => Self::RelativeAngularAcceleration,
            SpectrumResponse::AbsoluteAcceleration => Self::AbsoluteAngularAcceleration,
            SpectrumResponse::PseudoVelocity => Self::PseudoAngularVelocity,
            SpectrumResponse::PseudoAcceleration => Self::PseudoAngularAcceleration,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSpectrumNodalField {
    quantity: ResponseSpectrumNodalQuantity,
    shape: Vec<usize>,
    values: Vec<f64>,
}

impl ResponseSpectrumNodalField {
    pub const fn quantity(&self) -> ResponseSpectrumNodalQuantity {
        self.quantity
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        if self.shape.len() != 2
            || self.shape[0] == 0
            || self.shape[1] != 3
            || self.shape[0].checked_mul(3) != Some(self.values.len())
            || self
                .values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(FeaRunError::Solve(
                "response-spectrum nodal field is not a finite component envelope".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSpectrumNodalEnvelope {
    translation: ResponseSpectrumNodalField,
    rotation: Option<ResponseSpectrumNodalField>,
}

impl ResponseSpectrumNodalEnvelope {
    pub const fn translation(&self) -> &ResponseSpectrumNodalField {
        &self.translation
    }

    pub const fn rotation(&self) -> Option<&ResponseSpectrumNodalField> {
        self.rotation.as_ref()
    }

    pub(super) fn validate_for(&self, response: SpectrumResponse) -> Result<(), FeaRunError> {
        self.translation.validate()?;
        if self.translation.quantity != ResponseSpectrumNodalQuantity::for_translation(response)
            || self.rotation.as_ref().is_some_and(|rotation| {
                rotation.validate().is_err()
                    || rotation.quantity != ResponseSpectrumNodalQuantity::for_rotation(response)
                    || rotation.shape != self.translation.shape
            })
        {
            return Err(FeaRunError::Solve(
                "response-spectrum nodal quantities do not match the spectrum response".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn recover_nodal_envelope(
    summary: &AssemblySummary,
    result: &ModalResponseSpectrumResult,
) -> Result<ResponseSpectrumNodalEnvelope, FeaRunError> {
    let projected = project_structural_nodal_snapshot(
        &summary.structural_dof_layout,
        result.combined_response(),
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    let shape = nodal_vector_shape(&summary.structural_dof_layout);
    Ok(ResponseSpectrumNodalEnvelope {
        translation: ResponseSpectrumNodalField {
            quantity: ResponseSpectrumNodalQuantity::for_translation(result.response()),
            shape: shape.clone(),
            values: projected.translations,
        },
        rotation: summary
            .structural_dof_layout
            .has_rotational_dofs()
            .then_some(ResponseSpectrumNodalField {
                quantity: ResponseSpectrumNodalQuantity::for_rotation(result.response()),
                shape,
                values: projected.rotations,
            }),
    })
}
