//! Real heat solve -> canonical source -> serialized native structural worker.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_linear_completion, prepare_semantic_state_objects,
    prepare_structural_sequence_step, AnalysisHostResponse, AnalysisOutputPolicy,
    StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::{
    fixtures::{fixture_model, FixtureId},
    solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
    ComputeBackend, ThermalProcedureControl, ThermalSolveOptions,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
const REFERENCE: f64 = 293.15;
const DELTA: f64 = 10.0;
const ALPHA: f64 = 1.0e-5;

fn model_and_mesh(order: ElementOrder, restrained: bool) -> (AnalysisModel, SolverMeshArtifact) {
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(REFERENCE),
    });
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(REFERENCE),
    });
    let thermal = model.materials[0].thermal.as_mut().unwrap();
    thermal.reference_temperature_k = runmat_types::physical::Temperature::new(REFERENCE);
    thermal.expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(ALPHA);
    let producer = ProcedureStep::thermal_steady("qualification_heat");
    let mut consumer = ProcedureStep::coupled(
        "qualification_expansion",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    );
    consumer.state_inputs.push(StateInput {
        input_id: "temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: producer.step_id.clone(),
            state_id: producer.state_outputs[0].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    model.sequence = AnalysisSequence::from_steps(
        "solved_temperature_patch",
        vec![producer.clone(), consumer.clone()],
    );
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    model.sequence.transitions.push(StateTransition {
        transition_id: TransitionId("temperature_patch_reuse".into()),
        producer_step: producer.step_id,
        consumer_step: consumer.step_id,
        state_id: producer.state_outputs[0].state_id.clone(),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
    let face = projection
        .boundary_faces
        .iter()
        .find(|face| {
            face.node_indices
                .iter()
                .all(|&node| mesh.topology.nodes[node].coordinates_m.canonical_value()[0] == 0.0)
        })
        .unwrap();
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "isothermal_wall".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(face.provenance[0].analysis_selection_id().unwrap()),
                value: PrescribedValue::Constant(Temperature::new(REFERENCE + DELTA)),
            },
        }));
    for node in &mesh.topology.nodes {
        // Exactly six zero components remove rigid modes without constraining dilation:
        // origin xyz, x-axis corner yz, y-axis corner z. All other DOFs are solved.
        let axes: &[usize] =
            if restrained || node.coordinates_m.canonical_value() == [0.0, 0.0, 0.0] {
                &[0, 1, 2]
            } else if node.coordinates_m.canonical_value() == [1.0, 0.0, 0.0] {
                &[1, 2]
            } else if node.coordinates_m.canonical_value() == [0.0, 1.0, 0.0] {
                &[2]
            } else {
                &[]
            };
        if axes.is_empty() {
            continue;
        }
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("qualification_node_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: axes
                        .iter()
                        .map(|&axis| TranslationDofValue {
                            dof: [
                                StructuralTranslationDof::Ux,
                                StructuralTranslationDof::Uy,
                                StructuralTranslationDof::Uz,
                            ][axis],
                            value: PrescribedValue::Constant(Displacement::new(0.0)),
                        })
                        .collect(),
                    rotations: Vec::new(),
                }),
            }));
    }
    (model, mesh)
}

fn revision() -> ProgramRevision {
    let digest = Digest::sha256(b"native-thermoelastic-chain");
    ProgramRevision::new(
        digest,
        digest,
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

#[test]
fn serialized_native_thermal_source_drives_tet4_and_tet10_static_fields_and_lineage() {
    for (order, restrained) in [
        (ElementOrder::Tet4, false),
        (ElementOrder::Tet10, false),
        (ElementOrder::Tet4, true),
        (ElementOrder::Tet10, true),
    ] {
        let (mut model, mesh) = model_and_mesh(order, restrained);
        model.sequence.steps[1].state_outputs.push(StateOutput {
            state_id: StateId("thermal-geometric".into()),
            kind: StateKind::GeometricStiffness,
            completeness: StateCompleteness::Complete,
            persistence: StatePersistence::UntilDependentsComplete,
        });
        let thermal = runmat_analysis_fea::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                procedure: ThermalProcedureControl::Steady,
                solver_mesh: Some(mesh.clone()),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        let state = CanonicalSemanticState::from_thermal_result(&model, &mesh, &thermal).unwrap();
        let state_bytes = state.canonical_bytes().unwrap();
        let limits = ObjectInventoryLimits::default();
        let digest = Digest::sha256(b"native-thermoelastic-chain");
        let mut prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
            authored_study_digest: digest,
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "study:thermal-native".into(),
                encryption_context: digest,
            },
            workload: StructuralLinearWorkload {
                schema_version: 2,
                step_id: model.sequence.steps[1].step_id.clone(),
                model: model.clone(),
                backend: ComputeBackend::Cpu,
                preconditioner: SpdPreconditionerKind::Jacobi,
                algebra_backend: LinearAlgebraBackendKind::CpuReference,
                observation_execution:
                    runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
                require_solver_mesh_for_solid: true,
                state_input: None,
            },
            mesh: Some(mesh.clone()),
            state_input: Some(
                prepare_semantic_state_objects(&state, 1024, ObjectNamespace::ResultValue, limits)
                    .unwrap(),
            ),
            chunk_bytes: 1024,
            inventory_limits: limits,
            required_capabilities: Default::default(),
            accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            solve_selection:
                runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                    machine_memory_bytes: None,
                },
        })
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
            directory.path().join("objects"),
            limits.max_object_bytes,
        )
        .unwrap();
        prepared.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &prepared.program.request,
            &mut store,
            &NativeAnalysisKernel,
            1024,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&prepared.host, &completed)
            .unwrap()
            .program_response();
        let imported =
            import_structural_linear_completion(&store, response.clone(), &prepared, limits)
                .unwrap();
        let invocation_index = prepared
            .inputs
            .iter()
            .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
            .unwrap();
        let original_invocation = prepared.inputs[invocation_index].clone();
        let mut foreign_workload =
            StructuralLinearWorkload::from_canonical_bytes(&original_invocation.payload()).unwrap();
        foreign_workload.model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k = runmat_types::physical::Quantity::new(
            foreign_workload.model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k
                .canonical_value()
                * (2.0),
        );
        prepared.inputs[invocation_index] = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SolverInvocation,
            original_invocation.manifest.logical_identity,
            &foreign_workload.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        assert!(
            import_structural_linear_completion(&store, response, &prepared, limits).is_err(),
            "completed thermal result cannot be reinterpreted under a foreign material model"
        );
        prepared.inputs[invocation_index] = original_invocation;
        let linearization = imported.result.linearization.as_ref().unwrap();
        let source = linearization.thermal_source().unwrap();
        source.validate_model(&model).unwrap();
        source.validate_result(&imported.result.run).unwrap();
        assert_eq!(source.transition().source_state, state.identity);
        assert_eq!(source.structural_result(), imported.identity);
        assert_eq!(imported.numerical_receipt.source_result, imported.identity);
        let displacement = imported
            .result
            .run
            .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        for (node, u) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
            for axis in 0..3 {
                let expected = if restrained {
                    0.0
                } else {
                    ALPHA * DELTA * node.coordinates_m.canonical_value()[axis]
                };
                assert!(
                    (u[axis] - expected).abs() < 1.0e-10,
                    "{order:?}: {u:?} expected {expected}"
                );
            }
        }
        assert_eq!(state.canonical_bytes().unwrap(), state_bytes);
        if restrained {
            let mechanical = model.materials[0].require_mechanical().unwrap();
            let pressure = mechanical.youngs_modulus_pa.canonical_value() * ALPHA * DELTA
                / (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value());
            let stress = imported
                .result
                .run
                .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_STRESS)
                .unwrap()
                .as_host_f64()
                .unwrap();
            assert_eq!(stress.len(), 6);
            assert!(stress[..3]
                .iter()
                .all(|value| (value / -pressure - 1.0).abs() < 1.0e-9));
            assert!(stress[3..]
                .iter()
                .all(|value| value.abs() < pressure * 1.0e-10));
            assert!(linearization
                .geometric_stiffness()
                .values()
                .iter()
                .any(|value| value.abs() > 1.0));
        }
        let recovery = runmat_analysis_execution::prepare_structural_field_recovery(
            runmat_analysis_execution::StructuralFieldRecoveryPreparation {
                source_host: prepared.host.clone(),
                source_invocation: prepared
                    .inputs
                    .iter()
                    .find(|input| {
                        input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation
                    })
                    .unwrap()
                    .clone(),
                source_result: imported.objects.clone(),
                mesh: prepared
                    .inputs
                    .iter()
                    .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::Mesh)
                    .cloned(),
                program_revision: revision(),
                chunk_bytes: 1024,
                inventory_limits: limits,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            },
        )
        .unwrap();
        recovery.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &recovery.program.request,
            &mut store,
            &NativeAnalysisKernel,
            1024,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&recovery.host, &completed)
            .unwrap()
            .program_response();
        let recovered = runmat_analysis_execution::import_structural_recovered_fields(
            &store, response, &recovery, limits,
        )
        .unwrap();
        assert_eq!(
            recovered.manifest.source,
            FieldRecoverySource::SolverResult(imported.identity)
        );
        let mut forged_recovery = recovery.inputs.clone();
        let invocation_input = forged_recovery
            .iter_mut()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
            .unwrap();
        *invocation_input = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SolverInvocation,
            invocation_input.manifest.logical_identity,
            &foreign_workload.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        assert!(
            NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: AnalysisWorkKind::RecoverField,
                    host: &recovery.host,
                    inputs: &forged_recovery,
                    output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
                })
                .is_err(),
            "recovery worker rejects a foreign material model"
        );
        let forged_request = FieldRecoveryRequest::new(
            state.producer,
            recovery.request.source,
            recovery.request.topology,
            recovery.request.plan.clone(),
        )
        .unwrap();
        let mut forged_recovery = recovery.inputs.clone();
        *forged_recovery
            .iter_mut()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::FieldRecoveryRequest)
            .unwrap() = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::FieldRecoveryRequest,
            forged_request.identity().digest(),
            &forged_request.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        let mut forged_host = recovery.host.clone();
        forged_host.workload.step = state.producer;
        assert!(
            NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: AnalysisWorkKind::RecoverField,
                    host: &forged_host,
                    inputs: &forged_recovery,
                    output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
                })
                .is_err(),
            "matching request/host forgery cannot rename the retained thermal result producer"
        );
        // Downstream consumers receive the same retained thermal authority, not a
        // mechanically reconstructed or independently reidentified linearization.
        let publication = runmat_analysis_execution::prepare_semantic_state_publication(
            runmat_analysis_execution::SemanticStatePublicationPreparation {
                source_host: prepared.host.clone(),
                source_result: imported.objects,
                source_profiles: imported.state_publication_profile_objects,
                source: SemanticStatePublicationSource::StructuralLinearization,
                declaration: model.sequence.steps[1]
                    .state_outputs
                    .last()
                    .unwrap()
                    .clone(),
                topology_digest: state.topology_digest,
                program_revision: revision(),
                chunk_bytes: 1024,
                inventory_limits: limits,
            },
        )
        .unwrap();
        publication.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &publication.program.request,
            &mut store,
            &NativeAnalysisKernel,
            1024,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&publication.host, &completed)
            .unwrap()
            .program_response();
        let published = runmat_analysis_execution::import_semantic_state(
            &store,
            response,
            &publication,
            limits,
        )
        .unwrap();
        assert_eq!(
            published
                .payload
                .structural_linearization()
                .unwrap()
                .thermal_source(),
            Some(source)
        );
        // Manifest substitution cannot authorize a different source.
        let mut foreign_host = prepared.host.clone();
        foreign_host.workload.step = state.producer;
        assert!(matches!(
            NativeAnalysisKernel.execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &foreign_host,
                inputs: &prepared.inputs,
                output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
            }),
            Err(AnalysisBridgeError::Identity(
                "structural task names a different authored procedure step"
            ))
        ));
        let mut wrong_inputs = prepared.inputs.clone();
        wrong_inputs
            .iter_mut()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SemanticState)
            .unwrap()
            .manifest
            .logical_identity = Digest::sha256(b"foreign-temperature");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &prepared.host,
                inputs: &wrong_inputs,
                output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
            })
            .is_err());
        // A different-sized, undecodable source must fail independent resource
        // rederivation first. It must never reach semantic field deserialization.
        let mut wrong_inputs = prepared.inputs.clone();
        let non_state_payload = prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
            .unwrap()
            .payload();
        let source_input = wrong_inputs
            .iter_mut()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SemanticState)
            .unwrap();
        *source_input = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SemanticState,
            state.identity.digest(),
            &non_state_payload,
            1024,
            ObjectNamespace::ResultValue,
            limits,
        )
        .unwrap();
        assert!(matches!(
            NativeAnalysisKernel.execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &prepared.host,
                inputs: &wrong_inputs,
                output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
            }),
            Err(AnalysisBridgeError::Identity(
                "resource plan does not converge from authoritative structural inputs"
            ))
        ));
    }
}
