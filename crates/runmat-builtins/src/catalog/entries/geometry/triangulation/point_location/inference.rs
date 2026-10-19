use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn double_array(rows: Option<usize>, columns: usize) -> ValueFact {
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![rows, Some(columns)]),
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    output
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let (rows, diagnostics) =
        super::super::query_facts::admit(request, "RM-CATALOG-POINTLOCATION", "pointLocation");
    let mut contract = CallContract::fixed(vec![double_array(rows, 1), double_array(rows, 3)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
