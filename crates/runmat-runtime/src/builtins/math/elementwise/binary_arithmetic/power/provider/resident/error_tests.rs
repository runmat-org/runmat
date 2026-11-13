use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mixed_publication_failure_retains_typed_cause_and_forbids_replay() {
    let source = runmat_accelerate_api::ProviderExecutionError::from_source(std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        "cast publication failed",
    ))
    .after_effect();
    let error = service_error(
        "mixed Single publication",
        runmat_accelerate_api::power_service::PowerServiceError(source),
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(!runmat_accelerate_api::is_unsupported_provider_error(
        &error
    ));
    let mut source: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(error) = source.downcast_ref::<std::io::Error>() {
            assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
            break;
        }
        source = source.source().expect("typed provider cause retained");
    }
}
