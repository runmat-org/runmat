//! Static electrical native state; potential is nodal, derived fields are physical element means.
use super::*;
use crate::execution::electrical_static::native_electrical_field_plan;
use crate::execution::{
    ElectricalStaticSolverResult, ElectricalStaticWorkload, FieldRecoveryRequest,
    FieldRecoverySource, FieldTopologyIdentity,
};
use crate::results::field::{
    RecoveredFieldArtifact, RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
    RecoveredFieldSample,
};
use runmat_analysis_core::{FieldLocation, StaticElectricalEquation};
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder, SolverMeshArtifact};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalElectricalStaticStatePayload {
    pub equation: StaticElectricalEquation,
    pub element_order: ElementOrder,
    pub ordered_node_ids: Vec<u64>,
    pub ordered_element_ids: Vec<u64>,
    pub recovery_request: FieldRecoveryRequest,
    pub fields: RecoveredFieldArtifact,
}
impl CanonicalElectricalStaticStatePayload {
    pub fn validate_for(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
        producer: StepIdentity,
    ) -> AnalysisExecutionResult<()> {
        self.recovery_request.validate()?;
        self.fields.validate().map_err(invalid)?;
        let plan = native_electrical_field_plan(
            self.equation,
            self.ordered_node_ids.len(),
            self.ordered_element_ids.len(),
        )?;
        let (flux, density) = match self.equation {
            StaticElectricalEquation::Electrostatics => (
                "electrical.electric_displacement",
                "electrical.electric_energy_density",
            ),
            StaticElectricalEquation::SteadyConduction => (
                "electrical.current_density",
                "electrical.joule_power_density",
            ),
        };
        let names = [
            "electrical.potential",
            "electrical.electric_field",
            flux,
            density,
        ];
        let unique = |ids: &[u64]| {
            !ids.is_empty()
                && !ids.contains(&0)
                && ids.iter().collect::<std::collections::BTreeSet<_>>().len() == ids.len()
        };
        if !matches!(self.element_order, ElementOrder::Tet4 | ElementOrder::Tet10)
            || !unique(&self.ordered_node_ids)
            || !unique(&self.ordered_element_ids)
            || self.recovery_request.plan != plan
            || self.fields.fields().len() != plan.requirements.len()
            || self.recovery_request.producer != producer
            || self.recovery_request.source != FieldRecoverySource::SolverResult(source_result)
            || self.fields.source != self.recovery_request.source
            || self.fields.topology_digest != topology_digest
            || !matches!(self.recovery_request.topology, FieldTopologyIdentity::VolumeMesh { mesh_digest }
                if Digest::from_bytes(*mesh_digest.bytes()) == topology_digest)
            || self.fields.recovery_request != self.recovery_request.identity()
            || self.fields.entity_counts
                != (crate::results::field::RecoveredFieldEntityCounts {
                    oriented_faces: 0,
                    nodes: self.ordered_node_ids.len(),
                    volume_elements: self.ordered_element_ids.len(),
                    volume_integration_points: 0,
                    oriented_edges: 0,
                    beam_elements: 0,
                    shell_elements: 0,
                    connectors: 0,
                    constraint_groups: 0,
                })
        {
            return Err(invalid(
                "electrical state equation/order/field inventory/source/topology differs",
            ));
        }
        for ((field, requirement), name) in self
            .fields
            .fields()
            .iter()
            .zip(&plan.requirements)
            .zip(names)
        {
            field_validation::validate_portable_field(field.storage())?;
            let definition = field.definition();
            let expected = crate::execution::electrical_static::native_electrical_field_definition(
                name,
                requirement,
            )?;
            if definition != &expected
                || field.sample() != RecoveredFieldSample::Final
                || definition.numeric_representation() != RecoveredFieldNumericRepresentation::Real
                || definition.derivation()
                    != crate::results::field::RecoveredFieldDerivation::Native
                || !matches!(
                    (definition.location(), definition.entity_domain()),
                    (FieldLocation::Node, RecoveredFieldEntityDomain::Nodes)
                        | (
                            FieldLocation::Element,
                            RecoveredFieldEntityDomain::VolumeElements
                        )
                )
            {
                return Err(invalid(
                    "electrical state field definition or native volume-average semantics differ",
                ));
            }
        }
        Ok(())
    }
}
impl CanonicalSemanticState {
    pub fn from_electrical_static_result(
        workload: &ElectricalStaticWorkload,
        mesh: &SolverMeshArtifact,
        result: &ElectricalStaticSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        mesh.validate_canonical().map_err(invalid)?;
        result.validate_for(workload, mesh)?;
        let requests = result.state_publication_requests()?;
        let [request] = requests.as_slice() else {
            return Err(invalid(
                "electrical state requires one persisted authored declaration",
            ));
        };
        Self::from_electrical_static_publication(request, result)
    }
    pub fn from_electrical_static_publication(
        request: &SemanticStatePublicationRequest,
        result: &ElectricalStaticSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_publication_request(request)?;
        let recovery_request = result.native_field_request_from_result()?;
        let fields = result.recover_native_fields_from_result(&recovery_request)?;
        let payload = CanonicalElectricalStaticStatePayload {
            equation: result.response().equation,
            element_order: result.element_order(),
            ordered_node_ids: result.response().ordered_node_ids.clone(),
            ordered_element_ids: result.response().ordered_element_ids.clone(),
            recovery_request,
            fields,
        };
        payload.validate_for(
            request.topology_digest,
            request.source_result,
            request.producer,
        )?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::ElectricalStatic {
                state: Box::new(payload),
            },
        })
    }
}
fn invalid(reason: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(reason.to_string())
}
