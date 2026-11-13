//! Native task admission contracts; no execution or physics qualification.
use super::*;
use runmat_analysis_execution::PreparedAnalysisTaskClosure;

#[test]
fn native_prepared_closure_binds_inputs_and_preserves_legacy_none_submission() {
    let mut study = ThermalStudyCoordinator::new(preparation(ElementOrder::Tet4, false)).unwrap();
    let task = study.prepare_solve().unwrap();
    let submit = |resource_plan, prepared_closure| {
        prepare_analysis_task(
            ExecutionScopeId::derive(&[b"native-closure-contract"]),
            PoolId::derive(&[b"native-closure-contract"]),
            &task.host,
            &task.program,
            Vec::new(),
            Default::default(),
            AnalysisTaskPolicy {
                resources: task.resource_plan.request.clone(),
                resource_plan,
                prepared_closure,
                retry: RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
                permitted_trust_tiers: crate::program_host::local_trust(),
            },
        )
    };
    let legacy = submit(Some(&task.resource_plan), None).unwrap();
    let bound = submit(
        Some(&task.resource_plan),
        Some(PreparedAnalysisTaskClosure {
            inputs: &task.inputs,
            output_policy: task.output_policy,
        }),
    )
    .unwrap();
    assert_eq!(legacy, bound);
    // The unplanned legacy path keeps the same original request and identity.
    assert_eq!(legacy, submit(None, None).unwrap());
    let mut changed = task.inputs.clone();
    changed.swap(0, 1);
    assert!(submit(
        Some(&task.resource_plan),
        Some(PreparedAnalysisTaskClosure {
            inputs: &changed,
            output_policy: task.output_policy,
        }),
    )
    .is_err());
    let no_plan = submit(
        None,
        Some(PreparedAnalysisTaskClosure {
            inputs: &task.inputs,
            output_policy: task.output_policy,
        }),
    )
    .unwrap_err();
    assert!(no_plan
        .to_string()
        .contains("prepared source closure requires an admitted resource plan"));
}
