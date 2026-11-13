use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DimensionFact,
    DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, StructFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) enum PiecewiseFamily {
    Pchip,
    Spline,
}

impl PiecewiseFamily {
    fn name(self) -> &'static str {
        match self {
            Self::Pchip => "pchip",
            Self::Spline => "spline",
        }
    }

    fn diagnostic(self) -> &'static str {
        match self {
            Self::Pchip => "RM-CATALOG-PCHIP-INPUT",
            Self::Spline => "RM-CATALOG-SPLINE-INPUT",
        }
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    family: PiecewiseFamily,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            family.diagnostic(),
            format!(
                "{} expects sample locations and values, with optional query points",
                family.name()
            ),
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, value) in request.arguments.iter().take(3).enumerate() {
        if !matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                family.diagnostic(),
                format!("{} inputs must be real numeric or logical", family.name()),
                index,
            ));
        }
    }

    let output = match request.arguments.len() {
        2 => polynomial_struct(),
        3 => evaluated_values(request),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

fn double_array() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Unknown,
        StorageFact::Dense,
    )
}

fn polynomial_struct() -> ValueFact {
    let mut fields = BTreeMap::new();
    fields.insert("form".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("breaks".into(), double_array());
    fields.insert("coefs".into(), double_array());
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    fields.insert("pieces".into(), scalar.clone());
    fields.insert("order".into(), scalar);
    // dim is scalar for scalar/vector-valued pp and a row vector for an
    // N-D component shape; Y's runtime shape decides which form is returned.
    let mut dim = ValueFact::unknown(DynamicReason::RuntimeValue);
    dim.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    fields.insert("dim".into(), dim);
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)))
}

fn evaluated_values(request: &CallRequest) -> ValueFact {
    let mut output = double_array();
    if request
        .arguments
        .get(1)
        .is_some_and(|sample| known_vector(&sample.shape))
    {
        output.shape = request.arguments[2].shape.clone();
        if output.shape.element_count() == Some(1) {
            output.storage = StorageFact::Scalar;
        }
    } else {
        // Nonscalar Y contributes component axes before the query axes. Its
        // chosen sample axis and extents are not proved from Xq alone.
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}

pub(super) fn known_vector(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Scalar => false,
        ShapeFact::Shaped { dims } => {
            dims.iter()
                .all(|dim| matches!(dim, DimensionFact::Known(_)))
                && dims
                    .iter()
                    .filter(|dim| matches!(dim, DimensionFact::Known(value) if *value > 1))
                    .count()
                    <= 1
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => false,
    }
}
