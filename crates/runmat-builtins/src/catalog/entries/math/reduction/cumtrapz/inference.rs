pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::super::numerical_inference::integration(
        request,
        entry,
        super::super::numerical_inference::IntegrationMode::Cumulative,
    )
}
