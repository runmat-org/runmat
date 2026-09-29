use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    if !(5..=6).contains(&args.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BLSPRICE-ARITY",
            "blsprice expects five inputs and an optional yield",
            args.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-BLSPRICE-OUTPUTS",
            "blsprice supports at most two outputs",
            0,
        ));
    }
    let mut shape = ShapeFact::Scalar;
    let mut all_host = true;
    for (index, value) in args.iter().enumerate() {
        if !matches!(
            value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || value
            .numeric()
            .is_some_and(|numeric| numeric.domain == NumericDomain::Complex)
            || value.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLSPRICE-INPUT",
                "pricing inputs must be real numeric or logical arrays",
                index,
            ));
        }
        if matches!(value.residency, ResidencyFact::Remote { .. }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLSPRICE-PLACEMENT",
                "distributed inputs are not supported",
                index,
            ));
        }
        all_host &= value.residency == ResidencyFact::Host;
        if index == 5 && value.shape.element_count() == Some(0) {
            continue;
        }
        shape = match broadcast_shape(&shape, &value.shape) {
            Ok(next) => next,
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-BLSPRICE-SHAPE",
                    "pricing inputs have incompatible implicit-expansion dimensions",
                    index,
                ));
                ShapeFact::Unknown
            }
        };
    }
    let mut output = if all_host {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape.clone(),
            if shape.element_count() == Some(1) {
                StorageFact::Scalar
            } else {
                StorageFact::Dense
            },
        )
    } else {
        // The provider may return single or double; fallback may materialize on host.
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.shape = shape.clone();
        fact
    };
    output.shape = shape;
    finish_fixed_outputs(entry, request, vec![output.clone(), output], diagnostics)
}
