//! Compiler admission composes original native closure residency with actual construction.
use super::*;
use runmat_analysis_fea::{
    constraints::{CanonicalStructuralConstraintPlan, ConstraintPlanError},
    planning::resource::{constraint_compilation_available_bytes, ByteCount},
};

#[test]
fn original_native_input_residency_precedes_compiler_boundary() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        let model = fixture::model(&mesh);
        let prepared = prepare(&model, &mesh);
        assert!(prepared
            .inputs
            .iter()
            .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan));
        let facts = prepared_input_resource_facts(&prepared.inputs).unwrap();
        let retained = facts.transport_closure_bytes.0
            + facts.decoded_resident_bytes_admission_budget.0
            + prepared.resource_plan.input.worker.fixed_memory_bytes.0;
        let reference = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            &model,
            Some(&mesh),
            u64::MAX,
        )
        .unwrap();
        let peak = reference.workspace().peak_requested_bytes();
        let compile = |ceiling| {
            let available = constraint_compilation_available_bytes(
                ceiling,
                &facts,
                prepared.resource_plan.input.worker.fixed_memory_bytes,
            )
            .unwrap();
            CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
                &model,
                Some(&mesh),
                available,
            )
        };
        assert_eq!(
            compile(retained + peak).unwrap().into_plan(),
            *reference.plan()
        );
        assert!(matches!(
            compile(retained + peak - 1),
            Err(ConstraintPlanError::WorkspaceLimit { .. })
        ));
        assert!(constraint_compilation_available_bytes(
            retained - 1,
            &facts,
            prepared.resource_plan.input.worker.fixed_memory_bytes,
        )
        .is_err());
        assert!(
            constraint_compilation_available_bytes(u64::MAX, &facts, ByteCount(u64::MAX)).is_err()
        );
    }
}

#[test]
fn cold_positive_understated_compiler_reservation_rejects_fresh_rederivation() {
    use runmat_analysis_execution::{
        prepare_analysis_objects, AnalysisKernel, AnalysisKernelInvocation, AnalysisOutputPolicy,
    };
    use runmat_analysis_fea::{
        execution::AnalysisWorkKind,
        planning::resource::{build_resource_plan, FeaResourceDemand},
    };
    use runmat_execution_artifact::ObjectNamespace;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        let model = fixture::model(&mesh);
        let prepared = prepare(&model, &mesh);
        let mut input = prepared.resource_plan.input.clone();
        let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut input.demand else {
            unreachable!()
        };
        assert!(demand.constraint_compilation_peak_bytes.0 > 1);
        demand.constraint_compilation_peak_bytes.0 -= 1;
        let understated = build_resource_plan(input).unwrap();
        assert_ne!(understated.identity, prepared.resource_plan.identity);
        let mut substituted = prepared.inputs.clone();
        let resource = substituted
            .iter_mut()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .unwrap();
        let limits = ObjectInventoryLimits::default();
        *resource = prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            understated.identity.digest(),
            &understated.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        let error = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
                kind: AnalysisWorkKind::Solve,
                host: &prepared.host,
                inputs: &substituted,
            })
            .unwrap_err();
        assert!(
            matches!(
                error,
                AnalysisBridgeError::Identity(
                    "resource plan does not converge from authoritative structural inputs"
                )
            ),
            "{error}"
        );
    }
}
