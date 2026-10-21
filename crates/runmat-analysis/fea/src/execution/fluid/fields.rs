//! Native field publication copies the original physical U/p/q; no solve or interpolation.
use super::*;
use crate::{
    execution::{
        FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoveryRequirement,
        FieldRecoverySource,
    },
    results::field::{
        FiniteVolumeFieldTopology, RecoveredField, RecoveredFieldArtifact,
        RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
        RecoveredFieldSample,
    },
};
use runmat_analysis_core::{AnalysisField, FieldLocation, ObservationQuantity, RecoveryPolicy};
use runmat_meshing_core::SolverMeshArtifact;
mod profile;

pub fn fluid_native_field_definition_inventory(
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    [
        (
            "cfd.velocity",
            ObservationQuantity::FluidVelocity,
            FieldLocation::Cell,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        (
            "cfd.pressure",
            ObservationQuantity::FluidPressure,
            FieldLocation::Cell,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        (
            "cfd.volume_flux",
            ObservationQuantity::VolumeFlowRate,
            FieldLocation::Face,
            RecoveredFieldEntityDomain::OrientedFaces,
        ),
    ]
    .into_iter()
    .map(|(id, quantity, location, domain)| {
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

/// Native fields of one workload: the flow fields, plus the conjugate node temperature when
/// the study couples heat transfer. Temperature is the solved nodal unknown, not a transfer.
pub fn field_definition_inventory(
    coupling: super::workload::FluidCoupling,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    let mut definitions = fluid_native_field_definition_inventory()?;
    if coupling == super::workload::FluidCoupling::FluidStructure {
        definitions.push(
            RecoveredFieldDefinition::new(
                "fsi.displacement",
                ObservationQuantity::Displacement,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationQuantity::Displacement.canonical_value_shape(),
                RecoveredFieldNumericRepresentation::Real,
            )
            .map_err(invalid)?,
        );
    }
    if coupling == super::workload::FluidCoupling::ConjugateHeat {
        definitions.push(
            RecoveredFieldDefinition::new(
                "cht.temperature",
                ObservationQuantity::Temperature,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationQuantity::Temperature.canonical_value_shape(),
                RecoveredFieldNumericRepresentation::Real,
            )
            .map_err(invalid)?,
        );
    }
    Ok(definitions)
}

fn plan(
    topology: &FiniteVolumeFieldTopology,
    coupling: super::workload::FluidCoupling,
) -> AnalysisExecutionResult<FieldRecoveryPlan> {
    let counts = topology.entity_counts();
    let requirements = field_definition_inventory(coupling)?
        .into_iter()
        .map(|d| {
            let n = u64::try_from(counts.entity_count(d.entity_domain())).map_err(invalid)?;
            let components = d
                .value_shape()
                .component_count()
                .ok_or_else(|| invalid("native fluid shape"))? as u64;
            Ok(FieldRecoveryRequirement {
                quantity: d.quantity(),
                source_location: d.location(),
                output_location: d.location(),
                recovery: RecoveryPolicy::Native,
                value_shape: d.value_shape().clone(),
                entity_domain: d.entity_domain(),
                cost: FieldRecoveryCost {
                    source_entity_count: n,
                    output_entity_count: n,
                    topology_incidence_count: 0,
                    arithmetic_operation_count: n
                        .checked_mul(components)
                        .ok_or_else(|| invalid("fluid field count overflow"))?,
                },
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let result = FieldRecoveryPlan { requirements };
    result.validate()?;
    Ok(result)
}

impl FluidSteadySolverResult {
    /// Source-derived requested heap for native construction; not measured execution usage.
    pub(crate) fn native_field_resident_bytes_estimate(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<u64> {
        profile::requested_bytes(self, workload, &workload.prepare(mesh)?)
    }
    pub fn native_field_topology(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FiniteVolumeFieldTopology> {
        // Validate without cloning retained residual/traction arrays.
        self.validate_for(workload, mesh)?;
        let source = workload.prepare(mesh)?;
        profile::admit(self, workload, &source)?;
        FiniteVolumeFieldTopology::derive(&source.source, &source.topology).map_err(invalid)
    }

    pub fn native_field_request(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        let topology = self.native_field_topology(workload, mesh)?;
        FieldRecoveryRequest::new(
            self.producer_identity(),
            FieldRecoverySource::SolverResult(self.identity()?),
            topology.identity(),
            plan(&topology, workload.coupling())?,
        )
    }

    pub fn recover_native_fields(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        let topology = self.native_field_topology(workload, mesh)?;
        let expected = FieldRecoveryRequest::new(
            self.producer_identity(),
            FieldRecoverySource::SolverResult(self.identity()?),
            topology.identity(),
            plan(&topology, workload.coupling())?,
        )?;
        if request != &expected {
            return Err(invalid(
                "fluid field request differs from original source/native layout",
            ));
        }
        let original = self.original();
        let velocity_count = original
            .velocity_m_per_s
            .len()
            .checked_mul(3)
            .ok_or_else(|| invalid("fluid velocity count overflow"))?;
        let mut velocity = Vec::with_capacity(velocity_count);
        velocity.extend(original.velocity_m_per_s.iter().flatten().copied());
        let mut values = vec![
            velocity,
            original.pressure_pa.clone(),
            original.face_volume_flux_m3_per_s.clone(),
        ];
        if let Some(energy) = self.conjugate_heat() {
            values.push(energy.temperature_k.clone());
        }
        if let Some(displacement) = self.coupled_displacement() {
            values.push(displacement.iter().flatten().copied().collect());
        }
        let mut values = values.into_iter();
        let definitions = field_definition_inventory(workload.coupling())?;
        let mut fields = Vec::with_capacity(definitions.len());
        for definition in definitions {
            let shape = definition
                .storage_shape(
                    topology
                        .entity_counts()
                        .entity_count(definition.entity_domain()),
                )
                .map_err(invalid)?;
            let storage = AnalysisField::host_f64(
                definition.field_id(),
                shape,
                values
                    .next()
                    .ok_or_else(|| invalid("fluid native inventory"))?,
            );
            fields.push(
                RecoveredField::new(definition, RecoveredFieldSample::Final, storage)
                    .map_err(invalid)?,
            );
        }
        request
            .plan
            .validate_definitions(fields.iter().map(|f| f.definition().clone()))?;
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            topology.identity().discretization_digest(),
            topology.entity_counts(),
            fields,
        )
        .map_err(invalid)
    }
}
