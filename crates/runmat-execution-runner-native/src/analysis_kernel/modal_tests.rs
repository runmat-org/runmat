use super::NativeAnalysisKernel;
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, IndependentDomain, PrescribedStructuralDofs,
    PrescribedValue, ProcedureStep, ProcedureStepId, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_analysis_execution::{
    execute_reserved_serial_analysis, import_structural_modal_result, prepare_analysis_objects,
    prepare_structural_modal_task, AnalysisHostResponse, AnalysisKernel, AnalysisKernelInvocation,
    AnalysisOutputPolicy, StructuralModalTaskPreparation,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, StructuralModalWorkload,
        StructuralModalWorkloadInput,
    },
    planning::resource::{build_resource_plan, FeaResourceDemand},
    results::observation::ObservationExecutionPolicy,
    ComputeBackend, ModalConvergencePolicy, ModalTruncationPolicy,
};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{
    cache::FilesystemObjectStore, object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace,
};
use runmat_meshing_core::ElementOrder;
use std::collections::BTreeSet;

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Tet4/Tet10 native scalar extraction, cold basis/receipt recovery and actual result
//   codec profiles check the shared output reservation. One-byte construction understatement
//   rejects. This finite workflow does not qualify arbitrary modal counts, realistic original
//   studies or measured peak memory.
#[test]
fn native_scalar_modal_preserves_cold_basis_receipts_fields_and_rejects_missing_source_budget() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut model, mesh) = super::structural_fixture::model_and_mesh(order);
        model.constraints = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                AnalysisConstraint::Structural(StructuralConstraint {
                    constraint_id: format!("scalar_support_{}", node.node_id),
                    frame: CoordinateFrame::Global,
                    kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                        target: RegionTarget::point(format!("node:{}", node.node_id)),
                        translations: [
                            StructuralTranslationDof::Ux,
                            StructuralTranslationDof::Uy,
                            StructuralTranslationDof::Uz,
                        ]
                        .into_iter()
                        .filter(|&dof| index == 0 || dof != StructuralTranslationDof::Ux)
                        .map(|dof| TranslationDofValue {
                            dof,
                            value: PrescribedValue::Constant(Displacement::new(0.)),
                        })
                        .collect(),
                        rotations: Vec::new(),
                    }),
                })
            })
            .collect();
        let mut step = ProcedureStep::structural_modal("modes");
        step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
        model.sequence.steps = vec![step];
        let workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model,
            step_id: ProcedureStepId("modes".into()),
            backend: ComputeBackend::Cpu,
            observation_execution: ObservationExecutionPolicy::scalable(),
            truncation: ModalTruncationPolicy::requested(1),
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            prestress_transition: None,
        })
        .unwrap();
        let limits = ObjectInventoryLimits::default();
        let prepared = prepare_structural_modal_task(StructuralModalTaskPreparation {
            local_execution:
                runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                    4_000,
                )
                .unwrap(),
            authored_study_digest: Digest::sha256(b"scalar-modal-study"),
            frozen_project_digest: None,
            program_revision: ProgramRevision::new(
                Digest::sha256(b"graph"),
                Digest::sha256(b"source"),
                ProgramEnvironment::new(
                    1,
                    1,
                    Digest::sha256(b"runtime"),
                    Digest::sha256(b"catalog"),
                    "matlab",
                )
                .unwrap(),
            )
            .unwrap(),
            artifact_access: ArtifactAccess {
                authorization_scope: "modal-test".into(),
                encryption_context: Digest::sha256(b"encryption"),
            },
            workload: workload.clone(),
            mesh: mesh.clone(),
            prestress_state: None,
            chunk_bytes: 256,
            inventory_limits: limits,
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
            solve_selection:
                runmat_analysis_fea::planning::resource::ModalSolveSelection::WithinCeiling {
                    machine_memory_bytes: None,
                },
        })
        .unwrap();
        // Original modal closures include the plan itself. Preserve the exact compiler
        // plan at the combined retained-input/construction boundary, rejecting minus one.
        let facts =
            runmat_analysis_execution::prepared_input_resource_facts(&prepared.inputs).unwrap();
        let retained = facts.transport_closure_bytes.0
            + facts.decoded_resident_bytes_admission_budget.0
            + prepared.resource_plan.input.worker.fixed_memory_bytes.0;
        let compiler = runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            workload.model(), Some(&mesh), u64::MAX,
        ).unwrap();
        let required = compiler.workspace().peak_requested_bytes();
        for extra in [required, required - 1] {
            let available =
                runmat_analysis_fea::planning::resource::constraint_compilation_available_bytes(
                    retained + extra,
                    &facts,
                    prepared.resource_plan.input.worker.fixed_memory_bytes,
                )
                .unwrap();
            let result = runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
                workload.model(), Some(&mesh), available,
            );
            if extra == required {
                assert_eq!(result.unwrap().into_plan(), *compiler.plan());
            } else {
                assert!(matches!(
                    result,
                    Err(
                        runmat_analysis_fea::constraints::ConstraintPlanError::WorkspaceLimit { .. }
                    )
                ));
            }
        }
        for compiler_substitution in [false, true] {
            let mut input = prepared.resource_plan.input.clone();
            let FeaResourceDemand::StructuralModalExtraction(demand) = &mut input.demand else {
                panic!("modal demand")
            };
            if compiler_substitution {
                assert!(demand.constraint_compilation_peak_bytes.0 > 1);
                demand.constraint_compilation_peak_bytes.0 -= 1;
            } else {
                assert!(demand.original_energy_source_bytes.0 > 0);
                demand.original_energy_source_bytes.0 = 0;
            }
            let understated = build_resource_plan(input).unwrap();
            let mut substituted = prepared.inputs.clone();
            let resource = substituted
                .iter_mut()
                .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
                .unwrap();
            *resource = prepare_analysis_objects(
                AnalysisNumericalObjectKind::ResourcePlan,
                understated.identity.digest(),
                &understated.canonical_bytes().unwrap(),
                256,
                ObjectNamespace::InputValue,
                limits,
            )
            .unwrap();
            assert!(NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    output_policy: AnalysisOutputPolicy::new(256, limits).unwrap(),
                    kind: AnalysisWorkKind::ExtractStructuralModes,
                    host: &prepared.host,
                    inputs: &substituted,
                })
                .is_err());
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("objects");
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        prepared.program.write_to(&mut store).unwrap();
        let completed = execute_reserved_serial_analysis(
            &prepared.program.request,
            &mut store,
            &NativeAnalysisKernel,
            256,
            limits,
            &prepared.resource_plan.request,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&prepared.host, &completed)
            .unwrap()
            .program_response();
        let hot =
            import_structural_modal_result(&store, response.clone(), &prepared, limits).unwrap();
        drop(store);
        let cold_store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let cold =
            import_structural_modal_result(&cold_store, response, &prepared, limits).unwrap();
        assert_eq!(cold.result, hot.result);
        assert_eq!(cold.numerical_receipt, hot.numerical_receipt);
        assert_eq!(
            cold.state_publication_profiles,
            hot.state_publication_profiles
        );
        assert_eq!(cold.result.basis().modes().len(), 1);
        let FeaResourceDemand::StructuralModalExtraction(demand) =
            &prepared.resource_plan.input.demand
        else {
            panic!("modal demand")
        };
        let actual = runmat_canonical_codec::profile_serialized(
            &cold.result,
            runmat_canonical_codec::CanonicalLimits::new(
                1024 * 1024 * 1024,
                20_000_000,
                65_536,
                64,
            ),
        )
        .unwrap();
        assert!(
            demand.result_output.encoded_payload_bytes_upper_bound.0
                >= actual.encoded_bytes_upper_bound
        );
        assert!(
            demand
                .result_output
                .construction_resident_bytes_admission_budget
                .0
                >= actual.materialization_resident_bytes_estimate
                    + demand.result_output.transport_closure_bytes_upper_bound.0
        );
        let mut understated = prepared.resource_plan.input.clone();
        let FeaResourceDemand::StructuralModalExtraction(changed) = &mut understated.demand else {
            unreachable!()
        };
        changed
            .result_output
            .construction_resident_bytes_admission_budget
            .0 -= 1;
        assert!(build_resource_plan(understated).is_err());
        assert!(cold.result.basis().modes()[0].frequency_hz() > 0.);
        cold.numerical_receipt
            .validate_against(cold.identity, &cold.result)
            .unwrap();
        let fields = runmat_analysis_fea::recover_completed_structural_modal_run(
            &workload,
            &mesh,
            &cold.result,
            &cold.numerical_receipt,
        )
        .unwrap();
        assert_eq!(fields.basis, *cold.result.basis());
        assert_eq!(
            fields.mode_shapes[0].shape,
            vec![mesh.topology.nodes.len(), 3]
        );
    }
}
