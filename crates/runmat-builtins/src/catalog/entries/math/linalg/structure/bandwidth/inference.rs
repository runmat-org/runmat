use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    if request.arguments.first().is_some_and(|fact| matches!(&fact.kind,
        ValueKindFact::Object(object) if object.runtime_class.as_ref() == Some(&standard::TRANSFER_FUNCTION.owned()))) {
        let mut diagnostics = Vec::<InferenceDiagnostic>::new();
        if request.arguments.len() > 2 || request.outputs.requested.known_count().is_some_and(|count| count > 1) {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-BANDWIDTH-TF-ARITY", "transfer-function bandwidth accepts at most one dB drop and one output", 1));
        }
        if let Some(drop) = request.arguments.get(1) {
            if !matches!(drop.kind, ValueKindFact::Unknown | ValueKindFact::Numeric(_)) {
                diagnostics.push(crate::catalog::inference::argument_error(
                    "RM-CATALOG-BANDWIDTH-TF-DROP", "transfer-function gain drop must be a real numeric scalar", 1));
            }
        }
        let mut output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double, domain: NumericDomain::Real,
        }));
        output.residency = ResidencyFact::Host;
        let mut contract = CallContract::fixed(vec![output]);
        contract.effects = entry.contract.effect_set();
        contract.capabilities = entry.contract.capability_set();
        let mut result = infer_call(&contract, request);
        result.diagnostics.extend(diagnostics);
        return result;
    }
    super::super::inference::infer(
        request,
        entry,
        super::super::inference::StructureOperation::Bandwidth,
    )
}
