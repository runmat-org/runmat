use super::*;

pub(super) fn execute(solved: &SolvedPipeline) -> RecoveredPipeline {
    let step = solved.prepared.host.workload.step;
    let mesh = SolverMeshArtifact::canonical_decode(&solved.prepared.inputs[1].payload()).unwrap();
    let topology =
        StructuralRecoveredFieldTopology::derive(&solved.solve.model, Some(&mesh)).unwrap();
    let request = FieldRecoveryRequest::new(
        step,
        FieldRecoverySource::SolverResult(solved.source_result),
        solved.constraint_plan.topology,
        runmat_analysis_fea::execution::FieldRecoveryPlan::resolve(
            solved.solve.procedure_step().unwrap(),
            topology,
            solved.constraint_plan.groups.len(),
        )
        .unwrap(),
    )
    .unwrap();
    let request = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        request.identity().digest(),
        &request.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let inputs = vec![
        request,
        solved.prepared.inputs[0].clone(),
        solved.solver_result.clone(),
        solved.prepared.inputs[1].clone(),
    ];
    let host = test_host_for_inputs(
        &solved.prepared.host,
        AnalysisWorkKind::RecoverField,
        &inputs,
        vec![AnalysisNumericalObjectKind::Field],
        Some(2),
    );
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::RecoverField,
            host: &host,
            inputs: &inputs,
        })
        .unwrap();
    let recovered_fields = prepare_kernel_output(&outputs[0]);
    let artifact = decode_recovered_field_artifact(&recovered_fields).unwrap();
    assert_eq!(
        artifact.source,
        FieldRecoverySource::SolverResult(solved.source_result)
    );
    assert!(artifact.fields().iter().any(|field| {
        field.storage().field_id == runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
    }));

    let mut substituted = inputs.clone();
    substituted[2].manifest.logical_identity = digest("substituted-mesh");
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::RecoverField,
            host: &host,
            inputs: &substituted,
        })
        .is_err());

    RecoveredPipeline {
        host,
        inputs,
        artifact,
    }
}
