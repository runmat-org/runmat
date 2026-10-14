use super::*;

#[test]
fn cleanup_unit_context_enters_only_its_callback_extent() {
    let base = RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let selected =
        base.clone()
            .with_language_policy(runmat_runtime::context::RuntimeLanguagePolicy {
                language_mode: runmat_runtime::context::RuntimeLanguageMode::Strict,
                extensions_enabled: false,
                top_level_await_enabled: base.top_level_await_enabled(),
                dynamic_eval_enabled: base.dynamic_eval_enabled(),
            });
    let job_id = JobId::from_bytes([7; 16]);
    let lease = SelectedGraphCleanupContexts {
        units: vec![base.clone(), selected],
        assignment: None,
        job_id: Some(job_id),
    };
    assert_eq!(base.execution_job(), None);
    let (factory_active, identity, mode) =
        futures::executor::block_on(lease.with_unit_context(1, |context| {
            let factory_active =
                runmat_runtime::context::legacy::active().map(|active| active.local_identity());
            async move {
                assert_eq!(context.execution_job(), Some(job_id));
                assert_eq!(
                    runmat_runtime::context::legacy::active().map(|active| active.local_identity()),
                    Some(context.local_identity())
                );
                (
                    factory_active,
                    context.local_identity(),
                    context.language_mode(),
                )
            }
        }))
        .expect("selected context");
    assert_eq!(factory_active, Some(base.local_identity()));
    assert_eq!(identity, base.local_identity());
    assert_eq!(mode, runmat_runtime::context::RuntimeLanguageMode::Strict);
    assert_eq!(base.execution_job(), None, "callback scope must retire");
    assert!(futures::executor::block_on(lease.with_unit_context(2, |_| async {})).is_err());
    assert_eq!(base.execution_job(), None);
}
