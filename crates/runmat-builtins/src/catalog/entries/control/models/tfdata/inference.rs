use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, CellFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-TFDATA-ARITY",
            "tfdata expects a model",
            0,
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 5)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TFDATA-OUTPUTS",
            "tfdata supports at most five outputs",
            0,
        ));
    }
    if let Some(model) = request.arguments.first() {
        let supported = match &model.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) => {
                    class == &standard::TRANSFER_FUNCTION.owned()
                        || class == &standard::STATE_SPACE.owned()
                }
                None => object.class.is_none(),
            },
            _ => false,
        };
        if !supported {
            diagnostics.push(argument_error(
                "RM-CATALOG-TFDATA-MODEL",
                "tfdata requires a tf or ss model or model array",
                0,
            ));
        }
    }
    if let Some(selector) = request.literals.literal_string_at(1) {
        if request.arguments.len() != 2 || !selector.eq_ignore_ascii_case("v") {
            diagnostics.push(argument_error(
                "RM-CATALOG-TFDATA-SELECTOR",
                "the only text selector is 'v' for a scalar SISO model",
                1,
            ));
        } else if request
            .arguments
            .first()
            .is_some_and(|model| model.shape.is_definitely_nonscalar())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-TFDATA-VECTOR-ARRAY",
                "the 'v' selector requires a scalar model",
                1,
            ));
        }
    }
    let numeric = ValueFact::unknown(DynamicReason::RuntimeValue);
    let vector = {
        let mut fact = numeric.clone();
        fact.shape = ShapeFact::from(vec![Some(1), None]);
        fact.storage = StorageFact::Dense;
        fact
    };
    let scalar_tf_only = request.arguments.first().is_some_and(|model| {
        matches!(
            &model.kind,
            ValueKindFact::Object(object)
                if object.runtime_class.as_ref() == Some(&standard::TRANSFER_FUNCTION.owned())
        ) && model.shape.element_count() == Some(1)
    });
    let array_result = request.arguments.len() == 1
        && request
            .arguments
            .first()
            .is_some_and(|model| model.shape.is_definitely_nonscalar());
    let coefficient_cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(vector.clone()),
            elements: if scalar_tf_only {
                vec![vector.clone()]
            } else {
                Vec::new()
            },
            elements_complete: scalar_tf_only,
        }),
        if array_result {
            ShapeFact::Unknown
        } else {
            ShapeFact::from(vec![
                if scalar_tf_only { Some(1) } else { None },
                if scalar_tf_only { Some(1) } else { None },
            ])
        },
        StorageFact::Dense,
    );
    let coefficient = match (
        request.arguments.len(),
        request.literals.literal_string_at(1),
    ) {
        (2, Some(selector)) if selector.eq_ignore_ascii_case("v") => vector,
        (1, _) => coefficient_cell.clone(),
        (count, None) if count > 2 => coefficient_cell,
        _ => numeric,
    };
    let sample_time = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let empty = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![
        coefficient.clone(),
        coefficient,
        sample_time,
        empty.clone(),
        empty,
    ]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
