//! Immutable initial-field admission shared by physical families.
//! Storage, field content identity and partition authority remain with recovered fields.
use runmat_analysis_core::{
    CoordinateFrame, FieldLocation, InitialCondition, InitialFieldArtifactRef, InitialValue,
    ObservationQuantity, ObservationValueShape,
};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldEntityDomain,
    RecoveredFieldNumericRepresentation,
};

use super::{AnalysisExecutionError, AnalysisExecutionResult};
mod selection;
pub(crate) use selection::{
    validate_declarations as validate_initial_field_declarations,
    validate_inventory as validate_initial_field_inventory,
    validate_selection as validate_initial_field_selection,
};

/// A closed source artifact, not permission to execute a consuming initial condition.
/// The consumer must additionally admit its quantity, frame and target topology.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "BindingWire")]
pub struct InitialFieldBinding {
    reference: InitialFieldArtifactRef,
    artifact: RecoveredFieldArtifact,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    reference: InitialFieldArtifactRef,
    artifact: RecoveredFieldArtifact,
}

impl TryFrom<BindingWire> for InitialFieldBinding {
    type Error = AnalysisExecutionError;

    fn try_from(wire: BindingWire) -> Result<Self, Self::Error> {
        Self::new(wire.reference, wire.artifact)
    }
}

fn invalid(detail: &str) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("initial field: {detail}"))
}

impl InitialFieldBinding {
    pub fn new(
        reference: InitialFieldArtifactRef,
        artifact: RecoveredFieldArtifact,
    ) -> AnalysisExecutionResult<Self> {
        let binding = Self {
            reference,
            artifact,
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn reference(&self) -> &InitialFieldArtifactRef {
        &self.reference
    }

    pub fn artifact(&self) -> &RecoveredFieldArtifact {
        &self.artifact
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.artifact
            .validate()
            .map_err(|error| invalid(&error.to_string()))?;
        let content = self
            .reference
            .content_digest
            .parse::<Digest>()
            .map_err(|_| invalid("declaration has no canonical content digest"))?;
        let topology = self
            .reference
            .topology_digest
            .parse::<Digest>()
            .map_err(|_| invalid("declaration has no canonical topology digest"))?;
        if self.reference.artifact_id.trim().is_empty()
            || content != self.artifact.identity().digest()
            || topology != self.artifact.topology_digest
        {
            return Err(invalid(
                "source content or topology differs from its declaration",
            ));
        }
        self.selected_field().map(|_| ())
    }

    fn selected_field(&self) -> AnalysisExecutionResult<&RecoveredField> {
        let mut fields = self
            .artifact
            .fields()
            .iter()
            .filter(|field| field.definition().field_id() == self.reference.field_id);
        let field = fields
            .next()
            .ok_or_else(|| invalid("declared field is absent"))?;
        if fields.next().is_some() {
            // InitialFieldArtifactRef has no sample selector. Do not silently choose a
            // history sample, interpolate it, or use the last value.
            return Err(invalid("declared field has ambiguous samples"));
        }
        Ok(field)
    }

    /// Borrow real SI-valued nodal content in the canonical target storage order.
    /// No topology transfer, frame rotation, device readback or numerical solve occurs here.
    pub fn admit<'a>(
        &'a self,
        condition: &InitialCondition,
        topology_digest: Digest,
        node_count: usize,
    ) -> AnalysisExecutionResult<AdmittedInitialField<'a>> {
        self.validate()?;
        let (reference, frame, quantity, shape) = requirement(condition)?;
        if reference != &self.reference || !matches!(frame, CoordinateFrame::Global) {
            return Err(invalid(
                "condition source or frame differs from the admitted field",
            ));
        }
        if topology_digest != self.artifact.topology_digest
            || node_count == 0
            || node_count != self.artifact.entity_counts.nodes
        {
            return Err(invalid("consumer topology or complete node count differs"));
        }
        let field = self.selected_field()?;
        let definition = field.definition();
        if definition.quantity() != quantity
            || definition.location() != FieldLocation::Node
            || definition.entity_domain() != RecoveredFieldEntityDomain::Nodes
            || !definition.value_shape().is_storage_equivalent_to(&shape)
            || definition.numeric_representation() != RecoveredFieldNumericRepresentation::Real
        {
            return Err(invalid(
                "quantity, nodal location, shape or real representation differs",
            ));
        }
        let values = field.storage().as_host_f64().ok_or_else(|| {
            invalid("field must be explicitly materialized on the execution host")
        })?;
        if values.iter().any(|value| {
            !value.is_finite() || (quantity == ObservationQuantity::Temperature && *value < 0.0)
        }) {
            return Err(invalid(
                "field has nonfinite or physically invalid initial values",
            ));
        }
        Ok(AdmittedInitialField {
            binding: self,
            field,
            values,
        })
    }
}

/// A validated borrow. The source field identity/sample is retained for consumer evidence.
#[derive(Clone, Copy, Debug)]
pub struct AdmittedInitialField<'a> {
    binding: &'a InitialFieldBinding,
    field: &'a RecoveredField,
    values: &'a [f64],
}

impl<'a> AdmittedInitialField<'a> {
    pub const fn binding(self) -> &'a InitialFieldBinding {
        self.binding
    }
    pub const fn field(self) -> &'a RecoveredField {
        self.field
    }
    pub const fn values(self) -> &'a [f64] {
        self.values
    }
}

fn requirement(
    condition: &InitialCondition,
) -> AnalysisExecutionResult<(
    &InitialFieldArtifactRef,
    &CoordinateFrame,
    ObservationQuantity,
    ObservationValueShape,
)> {
    fn source<Q>(
        value: &runmat_analysis_core::InitialConditionValue<Q>,
    ) -> AnalysisExecutionResult<(&InitialFieldArtifactRef, &CoordinateFrame)> {
        match &value.value {
            InitialValue::FieldArtifact(reference) => Ok((reference, &value.frame)),
            _ => Err(invalid("condition is not a field-artifact input")),
        }
    }
    use InitialCondition as C;
    use ObservationQuantity as Q;
    use ObservationValueShape as S;
    let ((reference, frame), quantity, shape) = match condition {
        C::Temperature(value) => (source(value)?, Q::Temperature, S::Scalar),
        C::Displacement(value) => (source(value)?, Q::Displacement, S::Vector3),
        C::Velocity(value) => (source(value)?, Q::Velocity, S::Vector3),
        C::ElectricPotential(value) => (source(value)?, Q::ElectricPotential, S::Scalar),
        C::MagneticVectorPotential(value) => {
            (source(value)?, Q::MagneticVectorPotential, S::Vector3)
        }
        C::AcousticPressure(value) => (source(value)?, Q::AcousticPressure, S::Scalar),
        C::FluidPressure(value) => (source(value)?, Q::FluidPressure, S::Scalar),
        C::FluidVelocity(value) => (source(value)?, Q::FluidVelocity, S::Vector3),
        C::AngularVelocity(_) => {
            return Err(invalid(
                "angular-velocity recovered-field quantity is not defined",
            ))
        }
    };
    Ok((reference, frame, quantity, shape))
}

#[cfg(test)]
mod tests;
