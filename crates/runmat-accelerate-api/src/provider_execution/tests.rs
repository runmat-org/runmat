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

#[test]
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

#[test]
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
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_error_retains_send_sync_boundary() {
    fn requires_send_sync<T: Send + Sync>() {}
    requires_send_sync::<ProviderExecutionError>();
}
