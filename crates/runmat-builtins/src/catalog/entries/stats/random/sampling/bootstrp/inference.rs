use super::super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // The Runtime zero-output form exits before argument/callback validation.
    if request.outputs.requested.known_count() == Some(0) {
        return finish(entry, request, Vec::new(), Vec::new(), false);
    }
    let diagnostics = arity(request, 3, None);
    let nboot = literal_count(request, 0);
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = ShapeFact::from(vec![nboot, None]);
    output.residency = ResidencyFact::Host;
    // Callback output class/width cannot be inferred from sample data class.
    let indices = numeric(
        ShapeFact::from(vec![None, nboot]),
        Some(NumericClass::Double),
        ResidencyFact::Host,
    );
    finish(entry, request, vec![output, indices], diagnostics, false)
}
