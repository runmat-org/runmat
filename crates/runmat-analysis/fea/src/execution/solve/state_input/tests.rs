use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;
use crate::execution::state::CanonicalThermalStatePayload;
use crate::execution::{
    CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
    CanonicalStateTransitionInput, SequenceIdentity, StepIdentity,
};
use runmat_analysis_core::*;
use runmat_execution::Digest;

fn workload(thermal: bool) -> StructuralLinearWorkload {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let producer = if thermal {
        ProcedureStep::thermal_steady("producer")
    } else {
        ProcedureStep::structural_static("producer")
    };
    let mut consumer = if thermal {
        ProcedureStep::coupled(
            "consumer",
            CoupledPhysicsFamily::Thermomechanical,
            AnalysisProcedure::Static,
        )
    } else {
        ProcedureStep::structural_static("consumer")
    };
    let state_kind = if thermal {
        StateKind::Thermal
    } else {
        StateKind::StructuralPrimary
    };
    let output = producer
        .state_outputs
        .iter()
        .find(|output| output.kind == state_kind)
        .unwrap();
    consumer.state_inputs.push(StateInput {
        input_id: "retained_source".into(),
        required_kind: state_kind,
        source: StateSource::Upstream {
            producer_step: producer.step_id.clone(),
            state_id: output.state_id.clone(),
        },
        use_as: if thermal {
            StateUse::Prestress
        } else {
            StateUse::InitialGuess
        },
        conflict: InitialStateConflictPolicy::Reject,
    });
    let authored = StateTransition {
        transition_id: TransitionId("exact_transfer".into()),
        producer_step: producer.step_id.clone(),
        consumer_step: consumer.step_id.clone(),
        state_id: output.state_id.clone(),
        method: if thermal {
            StateTransitionMethod::Reuse
        } else {
            StateTransitionMethod::InitialGuessOnly
        },
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    };
    model.sequence =
        AnalysisSequence::from_steps("typed_structural_state", vec![producer, consumer]);
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    model.sequence.transitions.push(authored);
    let mut result = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[1].step_id.clone(),
        model,
        backend: crate::ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution: ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let sequence = SequenceIdentity::from_solver_contract(&result.model.sequence).unwrap();
    let transition = bound_transition(&result, sequence, thermal);
    result.state_input = Some(if thermal {
        StructuralLinearStateInput::ThermalPrestress(transition)
    } else {
        StructuralLinearStateInput::InitialGuess(transition)
    });
    result
}

fn bound_transition(
    workload: &StructuralLinearWorkload,
    sequence: SequenceIdentity,
    thermal: bool,
) -> CanonicalStateTransition {
    let producer_contract = &workload.model.sequence.steps[0];
    let consumer_contract = &workload.model.sequence.steps[1];
    let authored = &workload.model.sequence.transitions[0];
    let producer = StepIdentity::from_solver_contract(sequence, producer_contract).unwrap();
    let consumer = StepIdentity::from_solver_contract(sequence, consumer_contract).unwrap();
    let declaration = producer_contract
        .state_outputs
        .iter()
        .find(|output| output.state_id == authored.state_id)
        .unwrap()
        .clone();
    let topology = Digest::sha256("shared solver topology");
    let payload = if thermal {
        CanonicalSemanticStatePayload::Thermal {
            state: Box::new(CanonicalThermalStatePayload {
                temperature: AnalysisField::host_f64("thermal.temperature", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![1], vec![303.15]),
                ordered_node_ids: vec![1],
                reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
                sample_time_s: runmat_types::physical::Time::new(0.0),
                maximum_relative_linear_residual: 0.0,
                interface_node_ids: Vec::new(),
                interface_temperature: None,
            }),
        }
    } else {
        CanonicalSemanticStatePayload::Fields {
            fields: vec![AnalysisField::host_f64(
                "structural.displacement", AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
                vec![1, 3],
                vec![0.0; 3],
            )],
        }
    };
    let source = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration,
        topology_digest: topology,
        source_result: SolverResultIdentity::from_step_result(
            producer,
            Digest::sha256("retained source result"),
        )
        .unwrap(),
        payload,
    })
    .unwrap();
    CanonicalStateTransition::new(CanonicalStateTransitionInput {
        sequence,
        producer,
        consumer,
        authored,
        source: &source,
        target_topology_digest: topology,
    })
    .unwrap()
}

#[test]
fn structural_state_input_roundtrip_and_projection_are_kind_specific() {
    for thermal in [false, true] {
        let workload = workload(thermal);
        workload.validate().unwrap();
        let decoded =
            StructuralLinearWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(decoded, workload);
        assert_eq!(decoded.solver_model().is_err(), thermal);
        let mut template = workload.clone();
        template.state_input = None;
        template.validate_sequence_template().unwrap();
        assert!(template.validate().is_err());
        assert!(workload.validate_sequence_template().is_err());
        let transition = workload.state_input.as_ref().unwrap().transition().clone();
        let mut wrong_kind = workload.clone();
        wrong_kind.state_input = Some(if thermal {
            StructuralLinearStateInput::InitialGuess(transition)
        } else {
            StructuralLinearStateInput::ThermalPrestress(transition)
        });
        assert!(wrong_kind.validate().is_err());
        let mut old_schema = workload.clone();
        old_schema.schema_version = 1;
        assert!(old_schema.validate().is_err());
        let mut obsolete_wire = serde_json::to_value(&workload).unwrap();
        let object = obsolete_wire.as_object_mut().unwrap();
        let state = object.remove("state_input").unwrap();
        object.insert("initial_guess_transition".into(), state);
        assert!(serde_json::from_value::<StructuralLinearWorkload>(obsolete_wire).is_err());
    }
}

#[test]
fn structural_state_input_rejects_valid_foreign_sequence_and_unsupported_authored_binding() {
    for thermal in [false, true] {
        let workload = workload(thermal);
        let foreign = bound_transition(
            &workload,
            SequenceIdentity::new(Digest::sha256("foreign sequence")).unwrap(),
            thermal,
        );
        foreign.validate().unwrap();
        let mut substituted = workload.clone();
        substituted.state_input = Some(if thermal {
            StructuralLinearStateInput::ThermalPrestress(foreign)
        } else {
            StructuralLinearStateInput::InitialGuess(foreign)
        });
        assert!(substituted.validate().is_err());
        let mut multiple = workload.clone();
        let mut duplicate = multiple.model.sequence.steps[1].state_inputs[0].clone();
        duplicate.input_id = "extra_source".into();
        multiple.model.sequence.steps[1]
            .state_inputs
            .push(duplicate);
        assert!(multiple.validate().is_err());
        let mut wrong_method = workload.clone();
        wrong_method.state_input = None;
        wrong_method.model.sequence.transitions[0].method = if thermal {
            StateTransitionMethod::InitialGuessOnly
        } else {
            StateTransitionMethod::Reuse
        };
        assert!(wrong_method.validate_sequence_template().is_err());
    }
    let mut wrong_family = workload(true);
    wrong_family.model.sequence.steps[1].family = PhysicsFamily::Structural;
    wrong_family.state_input = None;
    assert!(wrong_family.validate_sequence_template().is_err());
}
