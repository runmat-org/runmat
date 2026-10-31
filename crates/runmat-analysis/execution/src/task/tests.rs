use super::*;
use crate::*;
use runmat_analysis_core::{StateCompleteness, StateId, StateKind, StateOutput, StatePersistence};
use runmat_execution::{value::ValuePayload, ProgramRevision};
use runmat_execution_artifact::ObjectNamespace;

fn publication() -> (
    PreparedSemanticStatePublication,
    ProgramRevision,
    AnalysisOutputPolicy,
) {
    let mut p = crate::fluid_study::tests::preparation();
    let declaration = StateOutput {
        state_id: StateId("original".into()),
        kind: StateKind::Fluid,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::UntilDependentsComplete,
    };
    let mut compiled = p.product.compiled().clone();
    compiled.model.sequence.steps[0].state_outputs = vec![declaration.clone()];
    p.product =
        runmat_analysis_fea::study::plan_fluid_study_product(compiled, &p.mesh, p.policy).unwrap();
    let result = p.workload().execute(&p.mesh).unwrap();
    let result = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        result.identity().unwrap().digest(),
        &result.canonical_bytes().unwrap(),
        p.output_policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        p.output_policy.inventory_limits(),
    )
    .unwrap();
    let policy = p.output_policy;
    let revision = p.program_revision.clone();
    let mut study = FluidStudyCoordinator::new(p).unwrap();
    let solve = study.prepare_solve().unwrap();
    (
        prepare_fluid_state_publication(FluidStatePublicationPreparation {
            source_host: solve.host.clone(),
            source_inputs: solve.inputs.clone(),
            source_result: result,
            declaration,
            program_revision: revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: policy.inventory_limits(),
        })
        .unwrap(),
        revision,
        policy,
    )
}

fn context(host: &AnalysisHostWorkload, plan: &FeaResourcePlan) -> AnalysisExecutionContext {
    let mut resources = plan.request.clone();
    resources
        .required_capabilities
        .extend(host.workload.generic_capabilities().unwrap());
    AnalysisExecutionContext {
        host: crate::tests::contract_default_host(),
        scope_id: ExecutionScopeId::derive(&[b"closure"]),
        pool_id: PoolId::derive(&[b"closure"]),
        program_artifact_id: ArtifactId::derive(&[b"closure"]),
        resources,
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    }
}

fn submit(
    p: &PreparedSemanticStatePublication,
    revision: ProgramRevision,
    policy: AnalysisOutputPolicy,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<TaskSubmission> {
    // Rebind changed closures canonically, so rejection cannot rely on stale host digests.
    let descriptors = inputs
        .iter()
        .map(|i| AnalysisWorkloadInput {
            kind: i.manifest.kind,
            logical_identity: i.manifest.logical_identity,
            payload_digest: i.manifest.payload_digest,
            manifest_digest: i.root.digest,
            manifest_encoded_length: i.root.encoded_length,
            reference_kind: if i.manifest.kind == AnalysisNumericalObjectKind::SolverResult {
                ValueRefKind::ResultObject
            } else {
                ValueRefKind::DriverObject
            },
        })
        .collect();
    let w = &p.host.workload;
    let host = AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            w.study,
            w.step,
            w.execution_plan,
            w.kind,
            descriptors,
            w.outputs.clone(),
            w.required_capabilities.clone(),
        )?,
        p.host.artifact_access.clone(),
        p.host.output_policy,
    )?;
    let program = prepare_analysis_program(&host, revision, inputs)?;
    let roots = program
        .request
        .arguments
        .iter()
        .map(|v| match v {
            ValuePayload::Object(r) => (**r).clone(),
            _ => unreachable!(),
        })
        .collect();
    build_resource_planned_task_submission_with_closure(
        &host,
        &p.resource_plan,
        PreparedAnalysisTaskClosure {
            inputs,
            output_policy: policy,
        },
        roots,
        Vec::new(),
        BTreeSet::new(),
        context(&host, &p.resource_plan),
    )
}

#[test]
fn immutable_original_source_and_current_plan_are_distinct_authorities() {
    let (p, revision, policy) = publication();
    assert_eq!(
        p.inputs
            .iter()
            .filter(|i| i.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .count(),
        2
    );
    submit(&p, revision.clone(), policy, &p.inputs).unwrap();
    let mut duplicate = p.inputs.clone();
    duplicate.push(p.inputs[5].clone());
    assert!(submit(&p, revision.clone(), policy, &duplicate).is_err());
    let mut duplicate = p.inputs.clone();
    duplicate.push(p.inputs[2].clone());
    assert!(submit(&p, revision.clone(), policy, &duplicate).is_err());
    let mut missing = p.inputs.clone();
    missing.remove(5);
    assert!(submit(&p, revision.clone(), policy, &missing).is_err());
    let mut substituted = p.inputs.clone();
    substituted[2] = p.inputs[5].clone();
    assert!(submit(&p, revision.clone(), policy, &substituted).is_err());
    let mut unrelated = crate::fluid_study::tests::preparation();
    unrelated.policy.numerical.momentum_relaxation *= 0.5;
    unrelated.product = runmat_analysis_fea::study::plan_fluid_study_product(
        unrelated.product.compiled().clone(),
        &unrelated.mesh,
        unrelated.policy,
    )
    .unwrap();
    let mut unrelated = FluidStudyCoordinator::new(unrelated).unwrap();
    let unrelated_plan = unrelated.prepare_solve().unwrap().inputs[2].clone();
    let mut substituted = p.inputs.clone();
    substituted[2] = unrelated_plan;
    assert!(submit(&p, revision, policy, &substituted).is_err());
}
