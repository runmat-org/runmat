use super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 0, Some(2));
    // A legacy LCG snapshot can be restored, so State length is not a constant.
    // Seed uses uint64 when binary64 cannot represent the exact restored seed.
    let mut seed = ValueFact::unknown(DynamicReason::RuntimeValue);
    seed.shape = ShapeFact::Scalar;
    let mut state = ValueFact::unknown(DynamicReason::RuntimeValue);
    state.shape = ShapeFact::Ranked { rank: 2 };
    let fields = std::collections::BTreeMap::from([
        ("Type".into(), ValueFact::scalar(ValueKindFact::String)),
        ("Seed".into(), seed),
        ("State".into(), state),
    ]);
    let mut output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    output.residency = ResidencyFact::Host;
    finish(entry, request, vec![output], diagnostics, false)
}
