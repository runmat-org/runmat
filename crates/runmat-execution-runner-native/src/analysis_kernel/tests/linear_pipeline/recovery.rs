use super::*;

pub(super) fn execute(solved: &SolvedPipeline) -> RecoveredPipeline {
    let prepared = runmat_analysis_execution::prepare_structural_field_recovery(
        runmat_analysis_execution::StructuralFieldRecoveryPreparation {
            source_host: solved.prepared.host.clone(),
            source_invocation: solved.prepared.inputs[0].clone(),
            source_result: solved.solver_result.clone(),
            mesh: Some(solved.prepared.inputs[1].clone()),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
    )
    .unwrap();
    let host = prepared.host;
    let inputs = prepared.inputs;
    assert_eq!(
        prepared.resource_plan.input.accuracy_ceiling,
        solved.solve.procedure_step().unwrap().accuracy.resources
    );
    assert_eq!(
        prepared.resource_plan.input.structural_operator(),
        None,
        "publishing static fields must not invent an equilibrium solve"
    );
    assert_eq!(
        runmat_analysis_execution::build_prepared_structural_recovery_resource_plan(
            &prepared.request,
            &inputs,
            host.output_policy,
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        )
        .unwrap(),
        prepared.resource_plan
    );
    for substitution in 0..3 {
        let mut input = prepared.resource_plan.input.clone();
        let runmat_analysis_fea::planning::resource::FeaResourceDemand::StructuralFieldRecovery(
            demand,
        ) = &mut input.demand
        else {
            panic!("static field recovery demand")
        };
        match substitution {
            0 => demand.constraint_compilation_peak_bytes.0 -= 1,
            1 => demand.construction_workspace_bytes.0 -= 1,
            _ => demand.constraint_plan_digest = digest("foreign-constraint-plan"),
        }
        let forged = runmat_analysis_fea::planning::resource::build_resource_plan(input).unwrap();
        let mut substituted = inputs.clone();
        let resource = substituted
            .iter_mut()
            .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .unwrap();
        *resource = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            forged.identity.digest(),
            &forged.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let substituted_host = test_host_for_inputs(
            &host,
            AnalysisWorkKind::RecoverField,
            &substituted,
            vec![AnalysisNumericalObjectKind::Field],
            Some(2),
        );
        assert!(
            NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    output_policy: output_policy(),
                    kind: AnalysisWorkKind::RecoverField,
                    host: &substituted_host,
                    inputs: &substituted,
                })
                .is_err(),
            "positive rehashed demand substitution {substitution}"
        );
    }
    let mut missing = inputs.clone();
    missing.retain(|i| i.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan);
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
            kind: AnalysisWorkKind::RecoverField,
            host: &host,
            inputs: &missing,
        })
        .is_err());
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
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
    for field_id in [
        runmat_analysis_fea::FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
        runmat_analysis_fea::FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
    ] {
        assert_eq!(
            artifact
                .fields()
                .iter()
                .filter(|field| field.definition().field_id() == field_id)
                .count(),
            1,
            "constraint-group fields have one canonical recovery authority"
        );
    }

    let mut substituted = inputs.clone();
    substituted[2].manifest.logical_identity = digest("substituted-mesh");
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: output_policy(),
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
