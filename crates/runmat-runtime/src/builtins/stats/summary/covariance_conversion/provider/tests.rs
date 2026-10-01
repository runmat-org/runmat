use super::provider_is_unsupported;

#[test]
fn provider_fallback_requires_typed_unsupported_not_error_wording() {
    assert!(!provider_is_unsupported(&anyhow::anyhow!(
        "device failed while processing an unsupported request"
    )));
    assert!(provider_is_unsupported(
        &runmat_accelerate_api::unsupported_provider_operation("fixture unavailable")
    ));
}
