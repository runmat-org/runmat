use super::NativeAnalysisKernel;
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, FormulationRequirement, IndependentDomain,
    PrescribedStructuralDofs, PrescribedValue, ProcedureStep, ProcedureStepId, RegionTarget,
    StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_modal_result, prepare_analysis_objects,
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
            formulation_requirement: FormulationRequirement::Experimental,
            observation_execution: ObservationExecutionPolicy::scalable(),
            truncation: ModalTruncationPolicy::requested(1),
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            prestress_transition: None,
        })
        .unwrap();
        let limits = ObjectInventoryLimits::default();
        let prepared = prepare_structural_modal_task(StructuralModalTaskPreparation {
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
        })
        .unwrap();
        let mut input = prepared.resource_plan.input.clone();
        let FeaResourceDemand::StructuralModalExtraction(demand) = &mut input.demand else {
            panic!("modal demand")
        };
        assert!(demand.original_energy_source_bytes.0 > 0);
        demand.original_energy_source_bytes.0 = 0;
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
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("objects");
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        prepared.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &prepared.program.request,
            &mut store,
            &NativeAnalysisKernel,
            256,
            limits,
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
