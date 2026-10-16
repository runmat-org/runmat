use crate::catalog::inference::{finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    StorageFact, ValueFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
    output.shape = request
        .arguments
        .first()
        .map(|input| input.shape.clone())
        .unwrap_or(runmat_types::ShapeFact::Unknown);
    output.storage = StorageFact::Sparse;
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, Vec::new())
}
