//! A completed, topology-bound thermal solution, distinct from prescribed temperature profiles.

use runmat_analysis_core::{
    AnalysisField, AnalysisModel, AnalysisProcedure, PhysicsFamily, StateCompleteness, StateKind,
};
use runmat_canonical_codec::encode_contract;
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::{Deserialize, Serialize};

use crate::{
    contracts::{fea_thermal_temperature_field_id, FeaThermalRunResult},
    diagnostics::FeaDiagnosticSeverity,
};

use super::{
    field_validation::validate_portable_fields, AnalysisExecutionError, AnalysisExecutionResult,
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
    SemanticStatePublicationRequest, SemanticStatePublicationSource, SequenceIdentity,
    SolverResultIdentity, StepIdentity, STATE_LIMITS, STATE_PREFIX,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalThermalStatePayload {
    pub temperature: AnalysisField,
    pub ordered_node_ids: Vec<u64>,
    pub reference_temperature_k: f64,
    pub sample_time_s: f64,
    pub maximum_relative_linear_residual: f64,
}

impl CanonicalThermalStatePayload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        validate_portable_fields(std::slice::from_ref(&self.temperature))?;
        let values = self.temperature.as_host_f64().ok_or_else(|| {
            AnalysisExecutionError::Invalid("thermal semantic state must be host-backed".into())
        })?;
        let mut unique_nodes = std::collections::BTreeSet::new();
        if self.ordered_node_ids.is_empty()
            || self.ordered_node_ids.len() != values.len()
            || self.temperature.shape != vec![self.ordered_node_ids.len()]
            || !self
                .ordered_node_ids
                .iter()
                .all(|node| unique_nodes.insert(*node))
            || values.iter().any(|value| *value <= 0.0)
            || !self.reference_temperature_k.is_finite()
            || self.reference_temperature_k <= 0.0
            || !self.sample_time_s.is_finite()
            || !self.maximum_relative_linear_residual.is_finite()
            || self.maximum_relative_linear_residual < 0.0
        {
            return Err(AnalysisExecutionError::Invalid(
                "thermal state requires positive finite nodal temperatures, unique ordered nodes, and finite reference/time/residual"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl CanonicalSemanticStatePayload {
    pub fn thermal(&self) -> Option<&CanonicalThermalStatePayload> {
        match self {
            Self::Thermal { state } => Some(state),
            _ => None,
        }
    }
}

impl CanonicalSemanticState {
    /// Publishes the terminal field of a completed physical thermal solve. The producer and
    /// source-result identities are derived from the authored step and solver output, not supplied
    /// by a plot, checkpoint, or downstream caller.
    pub fn from_thermal_result(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        result: &FeaThermalRunResult,
    ) -> AnalysisExecutionResult<Self> {
        mesh.validate_canonical().map_err(|error| {
            AnalysisExecutionError::Invalid(format!("invalid thermal solver mesh: {error}"))
        })?;
        result.run.validate_solver_formulation().map_err(|error| {
            AnalysisExecutionError::Invalid(format!("invalid thermal solver result: {error}"))
        })?;
        if result.run.identity.family() != PhysicsFamily::Thermal {
            return Err(AnalysisExecutionError::Invalid(
                "thermal semantic state requires a physical thermal result".into(),
            ));
        }
        let sequence = SequenceIdentity::from_solver_contract(&model.sequence)?;
        let matching = model
            .sequence
            .steps
            .iter()
            .filter(|step| {
                step.step_id.0 == result.initial_state.procedure_step_id
                    && step.family == PhysicsFamily::Thermal
                    && step.procedure == result.run.identity.procedure()
            })
            .collect::<Vec<_>>();
        let [step] = matching.as_slice() else {
            return Err(AnalysisExecutionError::Invalid(
                "thermal result must match exactly one authored thermal producer step".into(),
            ));
        };
        let producer = StepIdentity::from_solver_contract(sequence, step)?;
        if result.producer != producer
            || !result.ordered_node_ids.iter().copied().eq(mesh
                .topology
                .nodes
                .iter()
                .map(|node| node.node_id))
        {
            return Err(AnalysisExecutionError::Identity(
                "thermal result producer or node ordering differs from the publication context",
            ));
        }
        if !matches!(
            step.procedure,
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse
        ) || result.initial_state.procedure_step_id != step.step_id.0
            || model
                .thermal
                .as_ref()
                .map(|domain| domain.reference_temperature.kelvin)
                != Some(result.reference_temperature_k)
            || result.initial_state.reference_temperature_k != result.reference_temperature_k
        {
            return Err(AnalysisExecutionError::Invalid(
                "thermal result reference or producer step differs from the authored study".into(),
            ));
        }
        let declarations = step
            .state_outputs
            .iter()
            .filter(|output| output.kind == StateKind::Thermal)
            .collect::<Vec<_>>();
        let [declaration] = declarations.as_slice() else {
            return Err(AnalysisExecutionError::Invalid(
                "thermal producer must declare exactly one thermal state output".into(),
            ));
        };
        if declaration.completeness != StateCompleteness::Complete {
            return Err(AnalysisExecutionError::Invalid(
                "thermal producer must declare complete solved state".into(),
            ));
        }
        if result.time_points_s.is_empty()
            || result.time_points_s.len() != result.temperature_snapshots.len()
            || result.residual_norms.len()
                != if step.procedure == AnalysisProcedure::TransientResponse {
                    result.time_points_s.len().saturating_sub(1)
                } else {
                    result.time_points_s.len()
                }
            || result.time_points_s.iter().any(|time| !time.is_finite())
            || result
                .time_points_s
                .windows(2)
                .any(|pair| pair[1] <= pair[0])
            || result
                .residual_norms
                .iter()
                .any(|residual| !residual.is_finite() || *residual < 0.0)
        {
            return Err(AnalysisExecutionError::Invalid(
                "thermal result has incomplete or nonmonotone terminal solve history".into(),
            ));
        }
        validate_portable_fields(&result.temperature_snapshots)?;
        if result
            .temperature_snapshots
            .iter()
            .enumerate()
            .any(|(index, field)| {
                field.field_id != fea_thermal_temperature_field_id(index)
                    || field.shape != vec![mesh.topology.nodes.len()]
                    || field
                        .as_host_f64()
                        .is_none_or(|values| values.iter().any(|value| *value <= 0.0))
            })
        {
            return Err(AnalysisExecutionError::Invalid(
                "thermal history has missing, nonphysical, or non-nodal temperature snapshots"
                    .into(),
            ));
        }
        for required in ["FEA_THERMAL_STABILITY", "FEA_THERMAL_ENERGY_BALANCE"] {
            if !result.run.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == required && diagnostic.severity == FeaDiagnosticSeverity::Info
            }) {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "thermal result lacks accepted {required} evidence"
                )));
            }
        }
        let terminal_index = result.temperature_snapshots.len() - 1;
        let temperature = result.temperature_snapshots[terminal_index].clone();
        let payload = CanonicalThermalStatePayload {
            temperature,
            ordered_node_ids: result.ordered_node_ids.clone(),
            reference_temperature_k: result.reference_temperature_k,
            sample_time_s: result.time_points_s[terminal_index],
            maximum_relative_linear_residual: result
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
        };
        payload.validate()?;
        let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
        if result.topology_digest != topology_digest {
            return Err(AnalysisExecutionError::Identity(
                "thermal result and state publication name different solver meshes",
            ));
        }
        let source_content = Digest::sha256(encode_contract(
            STATE_PREFIX,
            "thermal-solver-result-content",
            &(
                &result.run.identity,
                &result.time_points_s,
                &result.temperature_snapshots,
                &result.temperature_gradient_snapshots,
                &result.heat_flux_snapshots,
                &result.energy_storage_rate_snapshots,
                &result.residual_norms,
                result.reference_temperature_k,
                result.topology_digest,
                result.producer,
                &result.ordered_node_ids,
                &result.initial_state,
            ),
            STATE_LIMITS,
        )?);
        let source_result = SolverResultIdentity::from_step_result(producer, source_content)?;
        let request = SemanticStatePublicationRequest::new(
            SemanticStatePublicationSource::ThermalPrimaryField,
            producer,
            (*declaration).clone(),
            topology_digest,
            source_result,
        )?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration,
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Thermal {
                state: Box::new(payload),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        contracts::{ComputeBackend, ThermalProcedureControl, ThermalSolveOptions},
        fixtures::{fixture_model, FixtureId},
    };
    use runmat_analysis_core::{
        AnalysisConstraint, AnalysisSequence, CoupledPhysicsFamily, InitialStateConflictPolicy,
        PrescribedValue, ProcedureStep, ReferenceStatePolicy, RegionTarget,
        SequenceDependencyReason, StateInput, StateSource, StateTransition, StateTransitionMethod,
        StateUse, Temperature, ThermalConstraint, ThermalConstraintKind, ThermalDomain,
        ThermoMechanicalDomain, TopologyTransitionPolicy, TransitionId,
    };
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    fn solved_thermal() -> (AnalysisModel, SolverMeshArtifact, FeaThermalRunResult) {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        model.sequence = AnalysisSequence::from_steps(
            "thermal_state_sequence",
            vec![ProcedureStep::thermal_transient("thermal_step")],
        );
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let result = crate::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 4 },
                solver_mesh: Some(mesh.clone()),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        (model, mesh, result)
    }

    #[test]
    fn completed_thermal_result_publishes_immutable_topology_bound_terminal_state() {
        let (model, mesh, result) = solved_thermal();
        let state = CanonicalSemanticState::from_thermal_result(&model, &mesh, &result).unwrap();
        assert_eq!(state.kind, StateKind::Thermal);
        assert_eq!(state.topology_digest, result.topology_digest);
        let payload = state.payload.thermal().unwrap();
        assert_eq!(payload.temperature, result.temperature_snapshots[3]);
        assert_eq!(payload.sample_time_s, result.time_points_s[3]);
        assert_eq!(payload.ordered_node_ids.len(), mesh.topology.nodes.len());
        let decoded =
            CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(decoded, state);

        let mut reordered = result.clone();
        reordered.ordered_node_ids.swap(0, 1);
        assert!(matches!(
            CanonicalSemanticState::from_thermal_result(&model, &mesh, &reordered),
            Err(AnalysisExecutionError::Identity(_))
        ));
        let mut substituted_producer = result.clone();
        substituted_producer.producer = StepIdentity::new(
            SequenceIdentity::new(Digest::sha256("another-sequence")).unwrap(),
            Digest::sha256("another-step"),
        )
        .unwrap();
        assert!(matches!(
            CanonicalSemanticState::from_thermal_result(&model, &mesh, &substituted_producer),
            Err(AnalysisExecutionError::Identity(_))
        ));
        let mut changed_flux = result.clone();
        if let runmat_analysis_core::AnalysisFieldValues::HostF64(values) =
            &mut changed_flux.heat_flux_snapshots[0].values
        {
            values[0] += 1.0;
        }
        let changed_state =
            CanonicalSemanticState::from_thermal_result(&model, &mesh, &changed_flux).unwrap();
        assert_ne!(state.source_result, changed_state.source_result);
        assert_eq!(state.payload, changed_state.payload);

        let mut forged_result = result.clone();
        forged_result.topology_digest = Digest::sha256("another-mesh");
        assert!(matches!(
            CanonicalSemanticState::from_thermal_result(&model, &mesh, &forged_result),
            Err(AnalysisExecutionError::Identity(_))
        ));
        let mut partial = result.clone();
        partial
            .run
            .diagnostics
            .retain(|diagnostic| diagnostic.code != "FEA_THERMAL_ENERGY_BALANCE");
        assert!(CanonicalSemanticState::from_thermal_result(&model, &mesh, &partial).is_err());
        let mut wrong_reference = result.clone();
        wrong_reference.reference_temperature_k += 1.0;
        assert!(
            CanonicalSemanticState::from_thermal_result(&model, &mesh, &wrong_reference).is_err()
        );
        let mut incomplete_history = result.clone();
        incomplete_history.temperature_snapshots.pop();
        assert!(
            CanonicalSemanticState::from_thermal_result(&model, &mesh, &incomplete_history)
                .is_err()
        );
        let mut invalid_payload = state.clone();
        if let CanonicalSemanticStatePayload::Thermal { state } = &mut invalid_payload.payload {
            state.temperature.values = runmat_analysis_core::AnalysisFieldValues::HostF64(vec![
                    -1.0;
                    state
                        .ordered_node_ids
                        .len()
                ]);
        }
        assert!(invalid_payload.validate().is_err());
        let mut duplicate_nodes = state.clone();
        if let CanonicalSemanticStatePayload::Thermal { state } = &mut duplicate_nodes.payload {
            state.ordered_node_ids[1] = state.ordered_node_ids[0];
        }
        assert!(duplicate_nodes.validate().is_err());
    }

    #[test]
    fn steady_thermal_result_publishes_a_single_solve_snapshot() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        model.sequence = AnalysisSequence::from_steps(
            "steady_thermal_state_sequence",
            vec![ProcedureStep::thermal_steady("steady_thermal_state_step")],
        );
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let face_id = projection.boundary_faces[0].provenance[0]
            .analysis_selection_id()
            .unwrap();
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: "steady_temperature".into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(face_id),
                    value: PrescribedValue::Constant(Temperature::new(315.0)),
                },
            }));
        let result = crate::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                procedure: ThermalProcedureControl::Steady,
                solver_mesh: Some(mesh.clone()),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        let state = CanonicalSemanticState::from_thermal_result(&model, &mesh, &result).unwrap();
        assert_eq!(state.payload.thermal().unwrap().sample_time_s, 0.0);
        assert_eq!(result.residual_norms.len(), 1);
        assert_eq!(state.payload.thermal().unwrap().temperature.shape, vec![4]);
    }

    #[test]
    fn completed_thermal_state_binds_authored_static_thermoelastic_reuse() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        model.thermo_mechanical = Some(ThermoMechanicalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        let producer = ProcedureStep::thermal_transient("thermal_step");
        let mut consumer = ProcedureStep::coupled(
            "thermoelastic_step",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::Static,
        );
        consumer.state_inputs.push(StateInput {
            input_id: "solved_temperature".into(),
            required_kind: StateKind::Thermal,
            source: StateSource::Upstream {
                producer_step: producer.step_id.clone(),
                state_id: producer.state_outputs[0].state_id.clone(),
            },
            use_as: StateUse::Prestress,
            conflict: InitialStateConflictPolicy::Reject,
        });
        let consumer_step_id = consumer.step_id.clone();
        model.sequence = AnalysisSequence::from_steps(
            "thermal_to_thermoelastic",
            vec![producer.clone(), consumer],
        );
        model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
        model.sequence.transitions.push(StateTransition {
            transition_id: TransitionId("solved_temperature_reuse".into()),
            producer_step: producer.step_id,
            consumer_step: consumer_step_id.clone(),
            state_id: producer.state_outputs[0].state_id.clone(),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        });
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let result = crate::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 4 },
                solver_mesh: Some(mesh.clone()),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        let source = CanonicalSemanticState::from_thermal_result(&model, &mesh, &result).unwrap();
        let bound = super::super::BoundThermalStateTransfer::bind(
            &model,
            &consumer_step_id,
            &source,
            &mesh,
        )
        .unwrap();
        assert_eq!(
            bound.nodal_temperature_k,
            source
                .payload
                .thermal()
                .unwrap()
                .temperature
                .as_host_f64()
                .unwrap()
        );
        assert_eq!(bound.transition.source_state, source.identity);
        assert_eq!(bound.reference_temperature_k, 293.15);
        assert_eq!(bound.source_time_s, *result.time_points_s.last().unwrap());

        let mut wrong_reference = model.clone();
        wrong_reference
            .thermo_mechanical
            .as_mut()
            .unwrap()
            .reference_temperature = Temperature::new(300.0);
        assert!(super::super::BoundThermalStateTransfer::bind(
            &wrong_reference,
            &consumer_step_id,
            &source,
            &mesh,
        )
        .is_err());
        let other_mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        assert!(super::super::BoundThermalStateTransfer::bind(
            &model,
            &consumer_step_id,
            &source,
            &other_mesh,
        )
        .is_err());
        let mut missing_capability = model.clone();
        missing_capability.materials[0].thermal = None;
        assert!(super::super::BoundThermalStateTransfer::bind(
            &missing_capability,
            &consumer_step_id,
            &source,
            &mesh,
        )
        .is_err());
    }
}
