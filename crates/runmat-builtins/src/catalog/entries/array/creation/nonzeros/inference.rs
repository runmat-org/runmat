use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ResidencyFact, ShapeFact, StorageFact, ValueFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    output.shape = ShapeFact::from(vec![None, Some(1)]);
    output.storage = StorageFact::Dense;
    output.residency = if matches!(output.residency, ResidencyFact::Host) {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    finish_fixed(entry, request, output, Vec::new())
}
