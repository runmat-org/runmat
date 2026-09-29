use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, AliasFact, CallContract, CallInference, CallRequest, CertaintyFact,
    ContiguityFact, DynamicReason, LayoutFact, MutationFact, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLE-ARITY",
            "pole expects exactly one model",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let mut count = None;
    if let Some(model) = request.arguments.first() {
        let valid = match &model.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class)
                    if class == &standard::TRANSFER_FUNCTION.owned()
                        || class == &standard::STATE_SPACE.owned() =>
                {
                    let required: &[&str] = if class == &standard::STATE_SPACE.owned() {
                        &["A", "Ts"]
                    } else {
                        &[
                            "Numerator",
                            "Denominator",
                            "Ts",
                            "InputDelay",
                            "OutputDelay",
                        ]
                    };
                    if object.properties_complete
                        && required
                            .iter()
                            .any(|name| !object.properties.contains_key(*name))
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-POLE-MODEL",
                            "model is missing required properties",
                            0,
                        ));
                    }
                    count = if class == &standard::STATE_SPACE.owned() {
                        object
                            .properties
                            .get("A")
                            .and_then(|a| a.shape.known_dims())
                            .and_then(|dims| match dims.as_slice() {
                                [Some(rows), Some(cols)] if rows == cols => Some(*rows),
                                _ => None,
                            })
                    } else {
                        object
                            .properties
                            .get("Denominator")
                            .and_then(|v| (v.shape.element_count() == Some(1)).then_some(0))
                    };
                    true
                }
                None => object.class.is_none(),
                Some(_) => false,
            },
            _ => false,
        };
        if !valid
            || model.shape.element_count().is_some_and(|n| n != 1)
            || matches!(model.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLE-MODEL",
                "sys must be a scalar supported dynamic-system object",
                0,
            ));
        }
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = ShapeFact::from(vec![count, Some(1)]);
    output.storage = StorageFact::Dense;
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.residency = ResidencyFact::Host;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    if count == Some(0) {
        output.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        output.certainty = CertaintyFact::Proven;
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
