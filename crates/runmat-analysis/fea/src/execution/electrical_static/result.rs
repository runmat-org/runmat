//! Retained physical arrays with validated lineage, not an independent accuracy certificate.
use super::*;
use crate::{
    execution::SolverResultIdentity,
    physics::electrical::{CanonicalElectricalResponse, ElectricalConstitutiveResponse},
    topology::AnalysisProjection,
};
use runmat_analysis_core::{AnalysisProcedure, PhysicsFamily, SolutionSpace};

mod integrals;
mod sources;
mod terminals;
const PREFIX: &[u8] = b"runmat-electrical-static-result/v2\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ElectricalStaticSolverResult {
    data: ResultData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultData {
    source_workload: AnalysisWorkloadIdentity,
    producer: StepIdentity,
    publication_declaration: Option<runmat_analysis_core::StateOutput>,
    formulation: AnalysisRunIdentity,
    projection_digest: Digest,
    element_order: runmat_meshing_core::ElementOrder,
    relative_residual_target: f64,
    response: CanonicalElectricalResponse,
}

impl<'de> Deserialize<'de> for ElectricalStaticSolverResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Self {
            data: ResultData::deserialize(deserializer)?,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl ElectricalStaticSolverResult {
    pub(super) fn new(
        workload: &ElectricalStaticWorkload,
        mesh: &SolverMeshArtifact,
        formulation: AnalysisRunIdentity,
        response: CanonicalElectricalResponse,
    ) -> AnalysisExecutionResult<Self> {
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let result = Self {
            data: ResultData {
                source_workload: workload.identity()?,
                producer: workload.producer_identity()?,
                publication_declaration: workload.publication_declaration()?.cloned(),
                formulation,
                projection_digest: projection.projection_digest,
                element_order: mesh.resolved_request.element_order,
                relative_residual_target: workload.relative_residual_target,
                response,
            },
        };
        result.validate_for(workload, mesh)?;
        Ok(result)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let d = &self.data;
        let r = &d.response;
        r.nodal_outward_flux
            .require_kind(r.equation.conjugate_flux_kind())
            .map_err(invalid)?;
        for terminal in &r.terminals {
            terminal
                .outward_flux
                .require_kind(r.equation.conjugate_flux_kind())
                .map_err(invalid)?;
        }
        r.volume_source_work
            .require_kind(r.equation.work_kind())
            .map_err(invalid)?;
        r.terminal_work
            .require_kind(r.equation.work_kind())
            .map_err(invalid)?;

        let tolerance = d.relative_residual_target;
        if let Some(declaration) = &d.publication_declaration {
            crate::execution::SemanticStatePublicationRequest::new(
                crate::execution::SemanticStatePublicationSource::StaticElectricalNativeFields,
                d.producer,
                declaration.clone(),
                Digest::from_bytes(*r.mesh_digest.bytes()),
                SolverResultIdentity::from_step_result(d.producer, Digest::sha256([]))?,
            )?;
        }
        if d.formulation.family() != PhysicsFamily::Electromagnetic
            || d.formulation.realization()
                != &crate::admission::canonical_electrical_realization(
                    r.equation,
                    d.element_order,
                    match (&r.charge_sources, r.equation) {
                        (None, _) => {
                            crate::physics::electrical::CanonicalElectricalSource::SourceFree
                        }
                        (
                            Some(_),
                            runmat_analysis_core::StaticElectricalEquation::Electrostatics,
                        ) => crate::physics::electrical::CanonicalElectricalSource::VolumeCharge,
                        (
                            Some(_),
                            runmat_analysis_core::StaticElectricalEquation::SteadyConduction,
                        ) => crate::physics::electrical::CanonicalElectricalSource::BoundaryCurrent,
                    },
                )
            || d.formulation.procedure() != AnalysisProcedure::Static
            || d.formulation.solution_space() != SolutionSpace::Direct
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || tolerance >= 1.0
            || !within(r.linear_relative_residual, tolerance)
            || !r.linear_absolute_residual.is_finite()
            || r.linear_absolute_residual < 0.0
            || !within(r.relative_flux_imbalance, tolerance.max(1e-10) * 10.0)
            || !within(
                r.relative_energy_identity_error,
                tolerance.max(1e-10) * 10.0,
            )
        {
            return Err(invalid(
                "electrical result formulation or numerical closure is invalid",
            ));
        }
        let nodes = r.ordered_node_ids.len();
        let elements = r.ordered_element_ids.len();
        if !unique(&r.ordered_node_ids)
            || !unique(&r.ordered_element_ids)
            || r.potential_v.canonical_values().len() != nodes
            || r.potential_increment_v.canonical_values().len() != nodes
            || r.nodal_outward_flux.canonical_values().len() != nodes
            || r.element_electric_field_v_per_m.canonical_values().len() != elements
            || r.potential_v
                .canonical_values()
                .iter()
                .chain(r.nodal_outward_flux.canonical_values())
                .chain(r.potential_increment_v.canonical_values())
                .any(|x| !x.is_finite())
            || r.element_electric_field_v_per_m
                .canonical_values()
                .iter()
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err(invalid(
                "electrical field cardinality, ordering or finiteness is invalid",
            ));
        }
        let (flux, density, total) = match (&r.equation, &r.constitutive) {
            (
                StaticElectricalEquation::Electrostatics,
                ElectricalConstitutiveResponse::Electrostatics {
                    displacement_c_per_m2,
                    energy_density_j_per_m3,
                    total_energy_j,
                },
            ) => (
                displacement_c_per_m2.canonical_values(),
                energy_density_j_per_m3.canonical_values(),
                total_energy_j.canonical_values(),
            ),
            (
                StaticElectricalEquation::SteadyConduction,
                ElectricalConstitutiveResponse::SteadyConduction {
                    current_density_a_per_m2,
                    joule_density_w_per_m3,
                    total_joule_power_w,
                },
            ) => (
                current_density_a_per_m2.canonical_values(),
                joule_density_w_per_m3.canonical_values(),
                total_joule_power_w.canonical_values(),
            ),
            _ => {
                return Err(invalid(
                    "electrical equation and constitutive fields disagree",
                ))
            }
        };
        if flux.len() != elements
            || density.len() != elements
            || flux.iter().flatten().any(|x| !x.is_finite())
            || density
                .iter()
                .chain(std::iter::once(total))
                .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(invalid(
                "electrical constitutive field shape or physical integral is invalid",
            ));
        }
        terminals::validate(r)?;
        sources::validate(r)?;
        Ok(())
    }

    /// Revalidates source, producer, exact formulation and canonical topology. It does not
    /// reconstruct missing fields, assemble an operator, rerun the solve or certify accuracy.
    pub fn validate_for(
        &self,
        workload: &ElectricalStaticWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let d = &self.data;
        let r = &d.response;
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        if d.source_workload != workload.identity()?
            || d.producer != workload.producer_identity()?
            || d.publication_declaration.as_ref() != workload.publication_declaration()?
            || d.formulation != workload.formulation(mesh)?
            || r.equation != workload.equation()?
            || d.relative_residual_target != workload.relative_residual_target
            || r.mesh_digest != mesh.canonical_digest
            || d.projection_digest != projection.projection_digest
            || d.element_order != mesh.resolved_request.element_order
            || !r
                .ordered_node_ids
                .iter()
                .copied()
                .eq(projection.nodes.iter().map(|n| n.node_id))
            || !r
                .ordered_element_ids
                .iter()
                .copied()
                .eq(projection.elements.iter().map(|e| e.element_id))
        {
            return Err(invalid(
                "electrical result does not belong to the workload and mesh",
            ));
        }
        terminals::validate_for(r, workload, mesh, &projection)?;
        let step = crate::physics::electrical::validate_canonical_input(
            &workload.model,
            workload.step_id.as_str(),
            workload.relative_residual_target,
        )
        .map_err(invalid)?;
        let sources = crate::physics::electrical::sources::assemble(
            &workload.model,
            mesh,
            step,
            &projection,
            workload.sample_point(),
        )
        .map_err(invalid)?;
        if r.charge_sources != sources {
            return Err(invalid(
                "retained charge source disagrees with authored volume integration",
            ));
        }
        integrals::validate_for(r, &projection)
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "electrical-static-result",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "electrical-static-result", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    /// Source-bound cold import is the acceptance boundary. Bare decoding only checks shape.
    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        workload: &ElectricalStaticWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        let value = Self::from_canonical_bytes(bytes)?;
        value.validate_for(workload, mesh)?;
        Ok(value)
    }
    pub fn source_result_identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        SolverResultIdentity::derive(
            "electrical-static-result",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.data.source_workload
    }
    pub fn producer(&self) -> StepIdentity {
        self.data.producer
    }
    pub fn formulation(&self) -> &AnalysisRunIdentity {
        &self.data.formulation
    }
    /// Immutable retained numerical fields in SI. No recovery solve occurs on access.
    pub fn response(&self) -> &CanonicalElectricalResponse {
        &self.data.response
    }
    pub fn mesh_digest(&self) -> runmat_meshing_core::StableDigest {
        self.data.response.mesh_digest
    }
    pub fn element_order(&self) -> runmat_meshing_core::ElementOrder {
        self.data.element_order
    }
    pub(super) fn publication_declaration(&self) -> Option<&runmat_analysis_core::StateOutput> {
        self.data.publication_declaration.as_ref()
    }
}

fn within(value: f64, bound: f64) -> bool {
    value.is_finite() && value >= 0.0 && value <= bound
}
fn unique(ids: &[u64]) -> bool {
    !ids.is_empty()
        && !ids.contains(&0)
        && ids.iter().collect::<std::collections::BTreeSet<_>>().len() == ids.len()
}
