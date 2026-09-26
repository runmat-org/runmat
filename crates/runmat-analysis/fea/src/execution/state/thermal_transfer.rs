//! Governed one-way transfer of a solved nodal temperature field onto identical solid topology.

use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, ProcedureStepId,
    StateCompleteness, StateKind, StateTransitionMethod, StateUse,
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalSemanticState,
    CanonicalStateTransition,
};
use crate::{
    assembly::thermoelastic_load::assemble_solved_temperature_load,
    properties::compile_structural_property_map,
};

/// Temperatures in the receiving solver mesh's node order. The transition is part of the
/// physical input identity; this is not a visualization field or an initial solver guess.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundThermalStateTransfer {
    pub transition: CanonicalStateTransition,
    pub nodal_temperature_k: Vec<f64>,
    pub physical_nodal_load_n: Vec<f64>,
    pub solid_recovery: Vec<crate::assembly::SolidRecoveryElementSummary>,
    pub reference_temperature_k: f64,
    pub source_time_s: f64,
}

impl BoundThermalStateTransfer {
    pub fn bind(
        model: &AnalysisModel,
        consumer_step_id: &ProcedureStepId,
        source: &CanonicalSemanticState,
        target_mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        source.validate()?;
        target_mesh.validate_canonical().map_err(|error| {
            AnalysisExecutionError::Invalid(format!("invalid receiving solver mesh: {error}"))
        })?;
        if source.kind != StateKind::Thermal || source.completeness != StateCompleteness::Complete {
            return Err(AnalysisExecutionError::Invalid(
                "thermoelastic transfer requires a completed thermal state".into(),
            ));
        }
        let consumer = model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == *consumer_step_id)
            .ok_or(AnalysisExecutionError::Identity(
                "thermoelastic consumer step is absent from the authored sequence",
            ))?;
        if consumer.family != PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            || consumer.procedure != AnalysisProcedure::Static
        {
            return Err(AnalysisExecutionError::Invalid(
                "solved thermal-state transfer currently requires a static thermomechanical consumer"
                    .into(),
            ));
        }
        let reference_temperature_k = model
            .thermo_mechanical
            .as_ref()
            .ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "thermoelastic consumer has no authored reference temperature".into(),
                )
            })?
            .reference_temperature
            .kelvin;
        let source_payload = source.payload.thermal().ok_or_else(|| {
            AnalysisExecutionError::Invalid("thermal source has no typed temperature field".into())
        })?;
        if reference_temperature_k != source_payload.reference_temperature_k {
            return Err(AnalysisExecutionError::Invalid(
                "thermal producer and thermoelastic consumer use different reference temperatures"
                    .into(),
            ));
        }
        let target_topology_digest = Digest::from_bytes(*target_mesh.canonical_digest.bytes());
        let transition = CanonicalStateTransition::bind_upstream(
            model,
            consumer_step_id,
            source,
            StateUse::Prestress,
            target_topology_digest,
        )?;
        if transition.method != StateTransitionMethod::Reuse {
            return Err(AnalysisExecutionError::Invalid(
                "solved thermal field currently requires exact same-topology reuse".into(),
            ));
        }
        let values = source_payload.temperature.as_host_f64().ok_or_else(|| {
            AnalysisExecutionError::Invalid("thermal state field is not host-backed".into())
        })?;
        let by_node = source_payload
            .ordered_node_ids
            .iter()
            .copied()
            .zip(values.iter().copied())
            .collect::<BTreeMap<_, _>>();
        if by_node.len() != target_mesh.topology.nodes.len() {
            return Err(AnalysisExecutionError::Invalid(
                "thermal state and receiving mesh have different nodal inventories".into(),
            ));
        }
        let nodal_temperature_k = target_mesh
            .topology
            .nodes
            .iter()
            .map(|node| {
                by_node.get(&node.node_id).copied().ok_or_else(|| {
                    AnalysisExecutionError::Invalid(format!(
                        "receiving solver node {} has no solved temperature",
                        node.node_id
                    ))
                })
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        let properties =
            compile_structural_property_map(model, Some(target_mesh)).map_err(|error| {
                AnalysisExecutionError::Invalid(format!(
                    "thermoelastic receiving materials are incomplete: {error}"
                ))
            })?;
        let physical_nodal_load_n =
            assemble_solved_temperature_load(target_mesh, &properties, &nodal_temperature_k)
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let solid_recovery = crate::assembly::solid_recovery::build(
            target_mesh,
            &properties,
            None,
            Some(&nodal_temperature_k),
        )
        .map_err(|error| {
            AnalysisExecutionError::Invalid(format!("solid thermal recovery: {error:?}"))
        })?;
        Ok(Self {
            transition,
            nodal_temperature_k,
            physical_nodal_load_n,
            solid_recovery,
            reference_temperature_k,
            source_time_s: source_payload.sample_time_s,
        })
    }
}
