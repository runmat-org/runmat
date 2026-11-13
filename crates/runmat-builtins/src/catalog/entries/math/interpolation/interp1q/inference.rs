use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DimensionFact,
    DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTERP1Q-ARITY",
            "interp1q requires sample locations, values, and query points",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, value) in request.arguments.iter().take(3).enumerate() {
        if !matches!(
            value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTERP1Q-INPUT",
                "interp1q inputs must be numeric or logical",
                index,
            ));
        }
    }
    let output = if request.arguments.len() == 3 {
        output_fact(&request.arguments[1], &request.arguments[2])
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

fn output_fact(samples: &ValueFact, query: &ValueFact) -> ValueFact {
    let domain = match samples.kind {
        ValueKindFact::Numeric(NumericFact { domain, .. }) => domain,
        ValueKindFact::Logical => NumericDomain::Real,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let shape = output_shape(&samples.shape, &query.shape);
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if shape.element_count().is_some() {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain,
        }),
        shape,
        storage,
    );
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}

fn output_shape(samples: &ShapeFact, query: &ShapeFact) -> ShapeFact {
    if super::super::piecewise_inference::known_vector(samples) {
        return query.clone();
    }
    let (ShapeFact::Shaped { dims }, Some(query_count)) = (samples, query.element_count()) else {
        return ShapeFact::Unknown;
    };
    if dims.len() < 2 {
        return ShapeFact::Unknown;
    }
    let mut output = Vec::with_capacity(dims.len());
    output.push(Some(query_count));
    for dim in &dims[1..] {
        output.push(match dim {
            DimensionFact::Known(extent) => Some(*extent),
            DimensionFact::Symbolic(_) | DimensionFact::Unknown => None,
        });
    }
    ShapeFact::from(output)
}
