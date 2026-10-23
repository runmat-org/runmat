pub(super) fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    let mut result = super::super::max::infer_omit_nan(request, entry);
    for output in &mut result.outputs {
        output.residency = runmat_types::ResidencyFact::Host;
    }
    result
}
