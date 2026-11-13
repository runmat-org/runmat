//! A completed, topology-bound thermal solution, distinct from prescribed temperature profiles.

use runmat_analysis_core::{AnalysisField, AnalysisModel, PhysicsFamily, StateKind};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::{Deserialize, Serialize};

use crate::contracts::FeaThermalRunResult;

use super::{
    field_validation::validate_portable_fields, AnalysisExecutionError, AnalysisExecutionResult,
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
    SemanticStatePublicationRequest, SemanticStatePublicationSource, SequenceIdentity,
    StepIdentity,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalThermalStatePayload {
    pub temperature: AnalysisField,
    pub ordered_node_ids: Vec<u64>,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub sample_time_s: runmat_types::physical::Time,
    pub maximum_relative_linear_residual: f64,
    /// Mesh nodes of finite-conductance thermal interfaces; `temperature` holds the first
    /// side's temperature there and `interface_temperature` the second side's.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interface_node_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interface_temperature: Option<AnalysisField>,
}

impl CanonicalThermalStatePayload {
    /// (interface node id, second side's temperature) pairs; empty without a
    /// finite-conductance interface.
    pub fn interface_temperature_by_node(&self) -> AnalysisExecutionResult<Vec<(u64, f64)>> {
        let Some(field) = &self.interface_temperature else {
            return Ok(Vec::new());
        };
        let values = field.as_host_f64().ok_or_else(|| {
            AnalysisExecutionError::Invalid("thermal interface state must be host-backed".into())
        })?;
        Ok(self
            .interface_node_ids
            .iter()
            .copied()
            .zip(values.iter().copied())
            .collect())
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        validate_portable_fields(std::slice::from_ref(&self.temperature))?;
        match &self.interface_temperature {
            None if self.interface_node_ids.is_empty() => {}
            Some(field)
                if !self.interface_node_ids.is_empty()
                    && field.shape == vec![self.interface_node_ids.len()]
                    && field
                        .as_host_f64()
                        .is_some_and(|values| values.iter().all(|value| *value > 0.0)) =>
            {
                validate_portable_fields(std::slice::from_ref(field))?;
            }
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "thermal state interface temperatures do not match their nodes".into(),
                ))
            }
        }
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
            || !self.reference_temperature_k.canonical_value().is_finite()
            || self.reference_temperature_k.canonical_value() <= 0.0
            || !self.sample_time_s.canonical_value().is_finite()
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
    /// Context-bound publication of the retained terminal thermal solution.
    pub fn from_thermal_result(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        result: &FeaThermalRunResult,
    ) -> AnalysisExecutionResult<Self> {
        mesh.validate_canonical()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        result.validate_complete()?;
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
                "thermal result must match exactly one authored producer step".into(),
            ));
        };
        let producer = StepIdentity::from_solver_contract(sequence, step)?;
        if result.producer != producer
            || result.topology_digest != Digest::from_bytes(*mesh.canonical_digest.bytes())
            || !result.ordered_node_ids.iter().copied().eq(mesh
                .topology
                .nodes
                .iter()
                .map(|node| node.node_id))
            || !result.ordered_element_ids.iter().copied().eq(mesh
                .topology
                .volume_elements
                .iter()
                .map(|element| element.element_id))
        {
            return Err(AnalysisExecutionError::Identity(
                "thermal result provenance differs from publication context",
            ));
        }
        if model
            .thermal
            .as_ref()
            .map(|domain| domain.reference_temperature.canonical_value())
            != Some(result.reference_temperature_k.canonical_value())
        {
            return Err(AnalysisExecutionError::Invalid(
                "thermal result reference differs from authored study".into(),
            ));
        }
        let declarations = step
            .state_outputs
            .iter()
            .filter(|output| output.kind == StateKind::Thermal)
            .collect::<Vec<_>>();
        let [declaration] = declarations.as_slice() else {
            return Err(AnalysisExecutionError::Invalid(
                "thermal producer must declare exactly one thermal output".into(),
            ));
        };
        let request = SemanticStatePublicationRequest::new(
            SemanticStatePublicationSource::ThermalPrimaryField,
            producer,
            (*declaration).clone(),
            result.topology_digest,
            result.source_result_identity()?,
        )?;
        Self::from_thermal_publication(&request, result)
    }

    /// Publish a decoded completed thermal result, verifying the request against its retained
    /// producer, topology and numerical content rather than guessing the solver-result schema.
    pub fn from_thermal_publication(
        request: &SemanticStatePublicationRequest,
        result: &FeaThermalRunResult,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        result.validate_complete()?;
        if request.source != SemanticStatePublicationSource::ThermalPrimaryField
            || request.producer != result.producer
            || request.topology_digest != result.topology_digest
            || request.source_result != result.source_result_identity()?
        {
            return Err(AnalysisExecutionError::Identity(
                "thermal publication request names a different source, producer or topology",
            ));
        }
        let terminal_index = result.temperature_snapshots.len() - 1;
        let payload = CanonicalThermalStatePayload {
            temperature: result.temperature_snapshots[terminal_index].clone(),
            ordered_node_ids: result.ordered_node_ids.clone(),
            reference_temperature_k: runmat_types::physical::Temperature::new(
                result.reference_temperature_k.canonical_value(),
            ),
            sample_time_s: runmat_types::physical::Time::new(result.time_points_s[terminal_index]),
            maximum_relative_linear_residual: result
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            interface_node_ids: result.interface_node_ids.clone(),
            interface_temperature: result
                .interface_temperature_snapshots
                .get(terminal_index)
                .cloned(),
        };
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
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
        AnalysisConstraint, AnalysisProcedure, AnalysisSequence, CoupledPhysicsFamily,
        InitialStateConflictPolicy, PrescribedValue, ProcedureStep, ReferenceStatePolicy,
        RegionTarget, SequenceDependencyReason, StateInput, StateSource, StateTransition,
        StateTransitionMethod, StateUse, Temperature, ThermalConstraint, ThermalConstraintKind,
        ThermalDomain, ThermoMechanicalDomain, TopologyTransitionPolicy, TransitionId,
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
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 4 },
                solver_mesh: Some(mesh.clone()),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        (model, mesh, result)
    }

    #[test]
    fn result_identity_binds_every_retained_field_and_ignores_presentation() {
        let (_, _, result) = solved_thermal();
        let identity = result.source_result_identity().unwrap();
        let mut negative_zero = result.clone();
        negative_zero.time_points_s[0] = -0.0;
        assert!(negative_zero.validate_complete().is_err());
        assert!(negative_zero.canonical_bytes().is_err());
        assert!(negative_zero.source_result_identity().is_err());
        for kind in 0..4 {
            for sample in 0..result.time_points_s.len() {
                let mut changed = result.clone();
                let fields = match kind {
                    0 => &mut changed.temperature_snapshots,
                    1 => &mut changed.temperature_gradient_snapshots,
                    2 => &mut changed.heat_flux_snapshots,
                    _ => &mut changed.energy_storage_rate_snapshots,
                };
                let runmat_analysis_core::AnalysisFieldValues::HostF64(values) =
                    &mut fields[sample].values
                else {
                    panic!("expected retained host field")
                };
                values[0] += 0.25;
                assert_ne!(identity, changed.source_result_identity().unwrap());
            }
        }
        let mut presentation = result.clone();
        for diagnostic in &mut presentation.run.diagnostics {
            diagnostic.message = "alternate presentation".into();
        }
        assert_eq!(identity, presentation.source_result_identity().unwrap());
        let portable =
            FeaThermalRunResult::from_canonical_bytes(&result.canonical_bytes().unwrap()).unwrap();
        assert_eq!(identity, portable.source_result_identity().unwrap());
    }

    // witness: FEA-SHARED-SEQUENCES-001/complete_immutable_state_publication @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: A completed thermal result publishes an immutable topology-bound terminal state.
    #[test]
    fn completed_thermal_result_publishes_immutable_topology_bound_terminal_state() {
        let (model, mesh, result) = solved_thermal();
        let state = CanonicalSemanticState::from_thermal_result(&model, &mesh, &result).unwrap();
        assert_eq!(state.kind, StateKind::Thermal);
        assert_eq!(state.topology_digest, result.topology_digest);
        let payload = state.payload.thermal().unwrap();
        assert_eq!(payload.temperature, result.temperature_snapshots[3]);
        assert_eq!(
            payload.sample_time_s.canonical_value(),
            result.time_points_s[3]
        );
        assert_eq!(payload.ordered_node_ids.len(), mesh.topology.nodes.len());
        let decoded =
            CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(decoded, state);
        let portable =
            FeaThermalRunResult::from_canonical_bytes(&result.canonical_bytes().unwrap()).unwrap();
        assert_eq!(portable, result);
        let request = SemanticStatePublicationRequest::new(
            SemanticStatePublicationSource::ThermalPrimaryField,
            result.producer,
            model.sequence.steps[0].state_outputs[0].clone(),
            result.topology_digest,
            result.source_result_identity().unwrap(),
        )
        .unwrap();
        assert_eq!(
            CanonicalSemanticState::from_thermal_publication(&request, &portable).unwrap(),
            state
        );
        let mut wrong_request = request.clone();
        wrong_request.topology_digest = Digest::sha256("wrong-publication-topology");
        assert!(
            CanonicalSemanticState::from_thermal_publication(&wrong_request, &portable).is_err()
        );
        let mut missing_flux = result.clone();
        missing_flux.heat_flux_snapshots.pop();
        assert!(missing_flux.canonical_bytes().is_err());
        // Exercise decode validation, bypassing the validating encoder deliberately.
        let malformed = runmat_canonical_codec::encode_contract(
            b"runmat-thermal-solver-result/v1\0",
            "thermal-solver-result",
            &missing_flux,
            runmat_canonical_codec::CanonicalLimits::new(1024 * 1024, 100_000, 65_536, 64),
        )
        .unwrap();
        assert!(FeaThermalRunResult::from_canonical_bytes(&malformed).is_err());
        let mut bad_shape = result.clone();
        bad_shape.temperature_gradient_snapshots[0].shape = vec![3];
        assert!(bad_shape.canonical_bytes().is_err());
        let mut bad_initial = result.clone();
        bad_initial.initial_state.reference_node_count += 1;
        assert!(bad_initial.canonical_bytes().is_err());
        let mut wrong_elements = result.clone();
        wrong_elements.ordered_element_ids[0] += 1;
        assert!(
            CanonicalSemanticState::from_thermal_result(&model, &mesh, &wrong_elements).is_err()
        );
        let mut contradictory = result.clone();
        let evidence = contradictory
            .run
            .diagnostics
            .iter()
            .find(|item| item.code == "FEA_THERMAL_STABILITY")
            .unwrap()
            .clone();
        contradictory.run.diagnostics.push(evidence);
        // Presentation diagnostics are not numerical acceptance authority.
        assert!(contradictory.canonical_bytes().is_ok());

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
        assert!(CanonicalSemanticState::from_thermal_publication(&request, &changed_flux).is_err());

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
        assert!(CanonicalSemanticState::from_thermal_result(&model, &mesh, &partial).is_ok());
        partial.energy_balances.clear();
        assert!(CanonicalSemanticState::from_thermal_result(&model, &mesh, &partial).is_err());
        let mut wrong_reference = result.clone();
        wrong_reference.reference_temperature_k = runmat_types::physical::Quantity::new(
            wrong_reference.reference_temperature_k.canonical_value() + (1.0),
        );
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
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(face_id)),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(315.0)),
                },
            }));
        let result = crate::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                procedure: ThermalProcedureControl::Steady,
                solver_mesh: Some(mesh.clone()),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        let state = CanonicalSemanticState::from_thermal_result(&model, &mesh, &result).unwrap();
        assert_eq!(
            state
                .payload
                .thermal()
                .unwrap()
                .sample_time_s
                .canonical_value(),
            0.0
        );
        assert_eq!(result.residual_norms.len(), 1);
        assert_eq!(state.payload.thermal().unwrap().temperature.shape, vec![4]);
    }

    // witness: FEA-SHARED-SEQUENCES-001/distinct_completed_state_kinds @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: A completed thermal state binds authored static thermoelastic reuse; structural and
    //   fluid state kinds are separate.
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
            bound.temperature.nodal_temperature_k.as_slice(),
            source
                .payload
                .thermal()
                .unwrap()
                .temperature
                .as_host_f64()
                .unwrap()
        );
        assert_eq!(bound.transition.source_state, source.identity);
        assert_eq!(bound.reference_temperature_k.canonical_value(), 293.15);
        assert_eq!(
            bound.source_time_s.canonical_value(),
            *result.time_points_s.last().unwrap()
        );

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
