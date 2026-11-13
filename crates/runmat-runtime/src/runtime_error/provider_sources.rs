use super::*;
use runmat_accelerate_api::ProviderExecutionError;
use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
type Owner = std::sync::Arc<()>;
#[cfg(target_arch = "wasm32")]
type Owner = std::rc::Rc<()>;

#[derive(Debug)]
struct Retained(Owner);
impl std::fmt::Display for Retained {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("provider-owned failure")
    }
}
impl Error for Retained {}

#[test]
fn nested_language_error_keeps_public_contract_and_native_or_local_owner() {
    let owner = Owner::new(());
    let weak = Owner::downgrade(&owner);
    let inner = build_runtime_error("invalid covariance weights")
        .with_identifier("RunMat:cov:WeightVectorLengthMismatch")
        .with_builtin("cov")
        .with_source(Retained(owner))
        .build();
    let source = ProviderExecutionError::from_source(inner);
    let error = with_provider_execution_source(
        build_runtime_error("provider failed")
            .with_identifier("RunMat:cov:Internal")
            .build(),
        source,
    );
    assert_eq!(
        error.identifier(),
        Some("RunMat:cov:WeightVectorLengthMismatch")
    );
    assert_eq!(error.message(), "invalid covariance weights");
    assert_eq!(error.context.builtin.as_deref(), Some("cov"));
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    let retained = error.source().unwrap().source().unwrap().source().unwrap();
    assert_eq!(
        Owner::strong_count(&retained.downcast_ref::<Retained>().unwrap().0),
        1
    );
    assert!(weak.upgrade().is_some());
    drop(error);
    assert!(weak.upgrade().is_none());
}

#[test]
fn ordinary_provider_failure_is_not_a_message_based_gather_retry() {
    let source =
        ProviderExecutionError::from(anyhow::anyhow!("unsupported-looking provider failure"));
    assert!(!source.is_unsupported());
    let error = with_provider_execution_source(
        build_runtime_error(source.to_string())
            .with_identifier("RunMat:qr:Internal")
            .build(),
        source,
    );
    assert_eq!(error.identifier(), Some("RunMat:qr:Internal"));
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert!(error.source().is_some());
}
