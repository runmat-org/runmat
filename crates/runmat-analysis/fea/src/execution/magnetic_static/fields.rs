//! Native retained magnetic array publication: copies only, no operator or recovery rerun.
use super::*;
use crate::physics::magnetic::MagneticFieldSample;
use crate::{
    execution::{
        FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoveryRequirement,
        FieldRecoverySource,
    },
    results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};
use runmat_analysis_core::{AnalysisField, FieldLocation, ObservationQuantity, RecoveryPolicy};

#[derive(Clone, Copy)]
pub(in crate::execution) enum NativeMagneticValues<'a> {
    Circulation(&'a [f64]),
    Potential(&'a [MagneticFieldSample]),
    Flux(&'a [MagneticFieldSample]),
    Intensity(&'a [MagneticFieldSample]),
}
impl<'a> NativeMagneticValues<'a> {
    pub(in crate::execution) fn values(self) -> impl Iterator<Item = f64> + 'a {
        let count = match self {
            Self::Circulation(values) => values.len(),
            Self::Potential(values) | Self::Flux(values) | Self::Intensity(values) => {
                values.len() * 3
            }
        };
        (0..count).map(move |i| match self {
            Self::Circulation(v) => v[i],
            Self::Potential(v) => v[i / 3].vector_potential_wb_per_m[i % 3],
            Self::Flux(v) => v[i / 3].flux_density_t[i % 3],
            Self::Intensity(v) => v[i / 3].magnetic_field_a_per_m[i % 3],
        })
    }
}
impl Serialize for NativeMagneticValues<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.values())
    }
}
pub(in crate::execution) struct NativeMagneticField<'a> {
    pub field_id: &'static str,
    pub values: NativeMagneticValues<'a>,
    pub shape: Vec<usize>,
}
pub(in crate::execution) fn native_magnetic_fields(
    result: &MagnetostaticSolverResult,
) -> [NativeMagneticField<'_>; 4] {
    let response = result.response();
    let count = response.samples.len();
    [
        NativeMagneticField {
            field_id: "magnetic.edge_circulation",
            values: NativeMagneticValues::Circulation(&response.edge_circulations_wb),
            shape: vec![response.edge_circulations_wb.len()],
        },
        NativeMagneticField {
            field_id: "magnetic.vector_potential",
            values: NativeMagneticValues::Potential(&response.samples),
            shape: vec![count, 3],
        },
        NativeMagneticField {
            field_id: "magnetic.flux_density",
            values: NativeMagneticValues::Flux(&response.samples),
            shape: vec![count, 3],
        },
        NativeMagneticField {
            field_id: "magnetic.field_intensity",
            values: NativeMagneticValues::Intensity(&response.samples),
            shape: vec![count, 3],
        },
    ]
}
impl MagnetostaticSolverResult {
    pub fn native_field_request(
        &self,
        workload: &MagnetostaticWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate_for(workload, mesh)?;
        self.native_field_request_from_result()
    }
    pub fn native_field_request_from_result(
        &self,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate()?;
        FieldRecoveryRequest::new(
            self.producer(),
            FieldRecoverySource::SolverResult(self.source_result_identity()?),
            self.sampling_topology().identity().map_err(invalid)?,
            native_magnetic_field_plan(self)?,
        )
    }
    pub fn recover_native_fields(
        &self,
        workload: &MagnetostaticWorkload,
        mesh: &SolverMeshArtifact,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        self.validate_for(workload, mesh)?;
        self.recover_native_fields_from_result(request)
    }
    pub fn recover_native_fields_from_result(
        &self,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        request.validate()?;
        if request != &self.native_field_request_from_result()? {
            return Err(invalid("magnetic request differs from exact retained producer/source/sampling/native inventory"));
        }
        let fields = native_magnetic_fields(self)
            .into_iter()
            .zip(&request.plan.requirements)
            .map(|(field, requirement)| {
                RecoveredField::new(
                    native_magnetic_field_definition(field.field_id, requirement)?,
                    RecoveredFieldSample::Final,
                    AnalysisField::host_f64(
                        field.field_id,
                        field.shape,
                        field.values.values().collect(),
                    ),
                )
                .map_err(invalid)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        request
            .plan
            .validate_definitions(fields.iter().map(|f| f.definition().clone()))?;
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            request.topology.discretization_digest(),
            native_magnetic_entity_counts(self),
            fields,
        )
        .map_err(invalid)
    }
}
pub(in crate::execution) fn native_magnetic_entity_counts(
    result: &MagnetostaticSolverResult,
) -> RecoveredFieldEntityCounts {
    result.sampling_topology().entity_counts()
}
fn native_magnetic_field_plan(
    result: &MagnetostaticSolverResult,
) -> AnalysisExecutionResult<FieldRecoveryPlan> {
    let specs = [
        (
            ObservationQuantity::MagneticEdgeCirculation,
            FieldLocation::Edge,
            RecoveredFieldEntityDomain::OrientedEdges,
            result.response().edge_circulations_wb.len(),
        ),
        (
            ObservationQuantity::MagneticVectorPotential,
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
            result.response().samples.len(),
        ),
        (
            ObservationQuantity::MagneticFluxDensity,
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
            result.response().samples.len(),
        ),
        (
            ObservationQuantity::MagneticFieldIntensity,
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
            result.response().samples.len(),
        ),
    ];
    let requirements = specs
        .into_iter()
        .map(|(quantity, location, entity_domain, count)| {
            let count = u64::try_from(count).map_err(invalid)?;
            let components = if quantity == ObservationQuantity::MagneticEdgeCirculation {
                1
            } else {
                3
            };
            Ok(FieldRecoveryRequirement {
                quantity,
                source_location: location,
                output_location: location,
                recovery: RecoveryPolicy::Native,
                value_shape: quantity.canonical_value_shape(),
                entity_domain,
                cost: FieldRecoveryCost {
                    source_entity_count: count,
                    output_entity_count: count,
                    topology_incidence_count: 0,
                    arithmetic_operation_count: count
                        .checked_mul(components)
                        .ok_or_else(|| invalid("magnetic field count overflow"))?,
                },
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let plan = FieldRecoveryPlan { requirements };
    plan.validate()?;
    Ok(plan)
}
pub(in crate::execution) fn native_magnetic_field_definition(
    field_id: &str,
    requirement: &FieldRecoveryRequirement,
) -> AnalysisExecutionResult<RecoveredFieldDefinition> {
    RecoveredFieldDefinition::new(
        field_id,
        requirement.quantity,
        requirement.output_location,
        requirement.entity_domain,
        requirement.value_shape.clone(),
        RecoveredFieldNumericRepresentation::Real,
    )
    .map_err(invalid)
}

/// Exact native definition inventory without constructing any numerical arrays.
pub fn magnetic_native_field_definitions(
    request: &FieldRecoveryRequest,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    request.validate()?;
    if !matches!(
        request.topology,
        crate::execution::FieldTopologyIdentity::SampledVolumeMesh { .. }
    ) {
        return Err(invalid(
            "magnetic native fields require original sampled-volume topology",
        ));
    }
    let definitions = magnetic_native_field_definition_inventory()?;
    if request.plan.requirements.len() != definitions.len() {
        return Err(invalid(
            "magnetic native inventory must contain all four original fields",
        ));
    }
    let samples = request.plan.requirements[1].cost.output_entity_count;
    let edge_upper = (samples / 15)
        .checked_mul(6)
        .ok_or_else(|| invalid("magnetic topology count overflow"))?;
    if samples == 0
        || !samples.is_multiple_of(15)
        || request.plan.requirements[2..]
            .iter()
            .any(|r| r.cost.output_entity_count != samples)
        || request.plan.requirements[0].cost.output_entity_count > edge_upper
    {
        return Err(invalid(
            "magnetic field inventory is not one complete fifteen-point tetrahedral sampling space",
        ));
    }
    request
        .plan
        .requirements
        .iter()
        .zip(definitions)
        .map(|(requirement, definition)| {
            let quantity = definition.quantity();
            let location = definition.location();
            let domain = definition.entity_domain();
            let components = quantity
                .canonical_value_shape()
                .component_count()
                .ok_or_else(|| invalid("magnetic component count"))?
                as u64;
            if requirement.quantity != quantity
                || requirement.source_location != location
                || requirement.output_location != location
                || requirement.entity_domain != domain
                || requirement.recovery != RecoveryPolicy::Native
                || requirement.value_shape != quantity.canonical_value_shape()
                || requirement.cost.source_entity_count != requirement.cost.output_entity_count
                || requirement.cost.topology_incidence_count != 0
                || requirement.cost.arithmetic_operation_count
                    != requirement
                        .cost
                        .output_entity_count
                        .checked_mul(components)
                        .ok_or_else(|| invalid("magnetic field size overflow"))?
            {
                return Err(invalid(
                    "magnetic native field inventory differs from canonical shape",
                ));
            }
            Ok(definition)
        })
        .collect()
}

/// Native physical definitions independent of any solved result or source identity.
pub fn magnetic_native_field_definition_inventory(
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    [
        (
            ObservationQuantity::MagneticEdgeCirculation,
            "magnetic.edge_circulation",
            FieldLocation::Edge,
            RecoveredFieldEntityDomain::OrientedEdges,
        ),
        (
            ObservationQuantity::MagneticVectorPotential,
            "magnetic.vector_potential",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::MagneticFluxDensity,
            "magnetic.flux_density",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::MagneticFieldIntensity,
            "magnetic.field_intensity",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
    ]
    .into_iter()
    .map(|(quantity, id, location, domain)| {
        RecoveredFieldDefinition::new(
            id,
            quantity,
            location,
            domain,
            quantity.canonical_value_shape(),
            RecoveredFieldNumericRepresentation::Real,
        )
        .map_err(invalid)
    })
    .collect()
}

#[cfg(test)]
mod tests;
