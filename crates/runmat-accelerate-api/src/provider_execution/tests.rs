use super::*;

#[cfg(not(target_arch = "wasm32"))]
type Owner = std::sync::Arc<()>;
#[cfg(target_arch = "wasm32")]
type Owner = std::rc::Rc<()>;

#[derive(Debug)]
struct RetainedOwner(Owner);
impl fmt::Display for RetainedOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("retained provider owner")
    }
}
impl Error for RetainedOwner {}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn nested_error_retains_actual_owner_until_final_error_drop() {
    let owner = Owner::new(());
    let weak = Owner::downgrade(&owner);
    let inner = ProviderExecutionError::from_source(RetainedOwner(owner));
    let outer = ProviderExecutionError::from_source(inner);
    assert_eq!(outer.to_string(), "retained provider owner");
    let source = outer.source().unwrap().source().unwrap();
    assert!(source.downcast_ref::<RetainedOwner>().is_some());
    assert_eq!(
        Owner::strong_count(&source.downcast_ref::<RetainedOwner>().unwrap().0),
        1
    );
    assert!(weak.upgrade().is_some());
    drop(outer);
    assert!(weak.upgrade().is_none());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn completion_exposes_original_cause_and_never_authorizes_post_effect_replay() {
    let owner = Owner::new(());
    let weak = Owner::downgrade(&owner);
    let primary = ProviderExecutionError::from_source(RetainedOwner(owner));
    let completion = ProviderExecutionCompletionError::new(Some(primary), Vec::new());
    assert!(completion.primary().is_some());
    assert!(completion.cleanup_errors().is_empty());
    assert!(completion
        .source()
        .unwrap()
        .source()
        .unwrap()
        .is::<RetainedOwner>());
    let outer = ProviderExecutionError::from_source(completion).after_effect();
    assert!(!outer.is_unsupported());
    assert!(weak.upgrade().is_some());
    drop(outer);
    assert!(weak.upgrade().is_none());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsupported_marker_survives_wrapping_but_ordinary_failure_is_not_fallback() {
    assert!(
        ProviderExecutionError::from(crate::unsupported_provider_operation(
            "direct unsupported marker"
        ))
        .is_unsupported()
    );
    let unsupported = ProviderExecutionError::from(
        crate::unsupported_provider_operation("unsupported shape").context("provider execution"),
    );
    assert!(unsupported.is_unsupported());
    let wrapped = ProviderExecutionError::from_source(unsupported);
    assert!(wrapped.is_unsupported());
    assert!(!ProviderExecutionError::from(anyhow::anyhow!("execution failed")).is_unsupported());

    let staged = ProviderExecutionError::from_source(ProviderExecutionError::from(
        crate::unsupported_provider_operation("stage unsupported after upload"),
    ))
    .after_effect();
    assert!(!staged.is_unsupported());
    assert!(staged.source().unwrap().source().is_some());
    let wrapped_staged = ProviderExecutionError::from_source(staged);
    assert!(wrapped_staged.is_after_effect());
    assert!(!wrapped_staged.is_unsupported());

    #[cfg(not(target_arch = "wasm32"))]
    {
        let wrapped = anyhow::Error::new(
            ProviderExecutionError::from(crate::unsupported_provider_operation(
                "post-effect source retained through anyhow",
            ))
            .after_effect(),
        )
        .context("outer operation failed");
        assert!(!crate::is_unsupported_provider_operation(&wrapped));
        assert!(wrapped
            .chain()
            .any(|source| source.is::<crate::UnsupportedProviderOperation>()));
        assert!(ProviderExecutionError::from(wrapped).is_after_effect());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn legacy_unknown_effect_does_not_replay_or_erase_actual_post_effect() {
    let unsupported = ProviderExecutionError::from(crate::unsupported_provider_operation(
        "legacy implementation returned unsupported",
    ));
    assert!(unsupported.is_unsupported());
    let unknown = unsupported.with_unproven_effects();
    assert_eq!(unknown.disposition, FallbackDisposition::Unproven);
    assert!(!unknown.is_unsupported());
    assert!(unknown
        .source()
        .is_some_and(|cause| cause.is::<crate::UnsupportedProviderOperation>()));
    assert_eq!(
        ProviderExecutionError::from_source(unknown).disposition,
        FallbackDisposition::Unproven
    );

    let actual = ProviderExecutionError::from(crate::unsupported_provider_operation(
        "failure after provider effect",
    ))
    .after_effect()
    .with_unproven_effects();
    assert_eq!(actual.disposition, FallbackDisposition::AfterEffect);
    assert!(!ProviderExecutionError::from_source(actual).is_unsupported());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_error_retains_send_sync_boundary() {
    fn requires_send_sync<T: Send + Sync>() {}
    requires_send_sync::<ProviderExecutionError>();
}

#[test]
fn async_execution_boundary_retains_platform_local_source() {
    let owner = Owner::new(());
    let weak = Owner::downgrade(&owner);
    let mut future: ProviderExecutionFuture<'static, ()> =
        Box::pin(async move { Err(ProviderExecutionError::from_source(RetainedOwner(owner))) });
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let std::task::Poll::Ready(Err(error)) = future.as_mut().poll(&mut context) else {
        panic!("immediate provider failure must be ready");
    };
    drop(future);
    assert!(!error.is_unsupported());
    assert!(error.source().unwrap().is::<RetainedOwner>());
    assert!(weak.upgrade().is_some());
    drop(error);
    assert!(weak.upgrade().is_none());
}
