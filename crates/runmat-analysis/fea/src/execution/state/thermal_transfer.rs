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
use crate::properties::compile_structural_property_map;

/// Temperatures in the receiving solver mesh's node order. The transition is part of the
/// physical input identity; this is not a visualization field or an initial solver guess.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundThermalStateTransfer {
    pub transition: CanonicalStateTransition,
    pub temperature: SolvedTemperatureField,
    pub reference_temperature_k: runmat_types::physical::Temperature,
    pub source_time_s: runmat_types::physical::Time,
}

/// A solved temperature field on the receiving mesh. Nodes of a finite-conductance thermal
/// interface carry two temperatures; the elements of the interface's second side read the
/// second one.
#[derive(Clone, Debug, PartialEq)]
pub struct SolvedTemperatureField {
    /// Mesh node order; the first side's temperature at interface nodes.
    pub nodal_temperature_k: Vec<f64>,
    /// Second-side dofs and their temperatures (after the mesh nodes), when interfaces split
    /// nodes.
    sides: Option<(crate::physics::thermal_interface::InterfaceDofs, Vec<f64>)>,
}

impl SolvedTemperatureField {
    pub fn nodal(nodal_temperature_k: Vec<f64>) -> Self {
        Self {
            nodal_temperature_k,
            sides: None,
        }
    }

    /// The temperature an element reads at one of its nodes (mesh element and node order).
    pub fn at(&self, element: usize, node: usize) -> f64 {
        match &self.sides {
            Some((dofs, values)) => values[dofs.dof(element, node)],
            None => self.nodal_temperature_k[node],
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodal_temperature_k.len()
    }

    pub fn all_positive_finite(&self) -> bool {
        let finite = |value: &f64| value.is_finite() && *value > 0.0;
        self.nodal_temperature_k.iter().all(finite)
            && self
                .sides
                .as_ref()
                .is_none_or(|(_, values)| values.iter().all(finite))
    }
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
            .canonical_value();
        let source_payload = source.payload.thermal().ok_or_else(|| {
            AnalysisExecutionError::Invalid("thermal source has no typed temperature field".into())
        })?;
        if reference_temperature_k != source_payload.reference_temperature_k.canonical_value() {
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
        let interface = source_payload.interface_temperature_by_node()?;
        let temperature = if interface.is_empty() {
            SolvedTemperatureField::nodal(nodal_temperature_k)
        } else {
            let projection = crate::topology::AnalysisProjection::derive(target_mesh)
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            let (_, dofs) =
                crate::physics::thermal_interface::solid_conduction(&projection, &model.interfaces)
                    .map_err(AnalysisExecutionError::Invalid)?;
            if dofs.second().len() != interface.len()
                || dofs
                    .second()
                    .iter()
                    .zip(&interface)
                    .any(|(&(node, _), &(id, _))| projection.nodes[node].node_id != id)
            {
                return Err(AnalysisExecutionError::Invalid(
                    "thermal state interface nodes differ from the receiving model's interfaces"
                        .into(),
                ));
            }
            let values = dofs
                .join(
                    &nodal_temperature_k,
                    &dofs
                        .second()
                        .iter()
                        .zip(&interface)
                        .map(|(&(node, _), &(_, value))| (node, value))
                        .collect::<Vec<_>>(),
                )
                .map_err(AnalysisExecutionError::Invalid)?;
            SolvedTemperatureField {
                nodal_temperature_k,
                sides: Some((dofs, values)),
            }
        };
        let properties =
            compile_structural_property_map(model, Some(target_mesh)).map_err(|error| {
                AnalysisExecutionError::Invalid(format!(
                    "thermoelastic receiving materials are incomplete: {error}"
                ))
            })?;
        properties
            .validate_thermal_solid_capabilities()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        // Binding owns source alignment and admission only. The receiving assembly owns
        // physical load integration and recovery data; never construct a second unused copy.
        Ok(Self {
            transition,
            temperature,
            reference_temperature_k: runmat_types::physical::Temperature::new(
                reference_temperature_k,
            ),
            source_time_s: runmat_types::physical::Time::new(
                source_payload.sample_time_s.canonical_value(),
            ),
        })
    }
}
