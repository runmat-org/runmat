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
    let mut saw_single = false;
    let mut double_forced = false;
    let mut unknown_input = false;
    let mut unknown_double_geometry = false;
    for (index, value) in args.iter().enumerate() {
        // At runtime a known-empty host Yield array is replaced with the
        // default double scalar before class, type, and shape admission.
        if index == 5 && value.residency == ResidencyFact::Host {
            let empty_array_kind = matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Character
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
            );
            if empty_array_kind
                && value.storage != StorageFact::Sparse
                && value.shape.element_count() == Some(0)
            {
                continue;
            }
            if value.kind == ValueKindFact::Unknown && value.shape.element_count() == Some(0) {
                // Accepted empty array kinds normalize to scalar Yield, but
                // the unresolved kind is not itself a precision proof.
                unknown_input = true;
                continue;
            }
            if matches!(
                value.kind,
                ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Cell(_)
            ) && value.shape.element_count().is_none()
            {
                // This value can be accepted only when runtime proves it is
                // empty. Its class therefore cannot be used to prove a result.
                unknown_input = true;
                continue;
            }
        }
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
        match &value.kind {
            ValueKindFact::Numeric(numeric) if numeric.class == NumericClass::Single => {
                saw_single = true;
            }
            ValueKindFact::Numeric(numeric) if numeric.class == NumericClass::Double => {
                if value.shape.element_count() != Some(1) {
                    if value.shape.element_count().is_some() {
                        double_forced = true;
                    } else {
                        unknown_double_geometry = true;
                    }
                }
            }
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
                if index == 5
                    && value.residency == ResidencyFact::Host
                    && value.shape.element_count().is_none() =>
            {
                // An empty host Yield normalizes to the default double
                // scalar; a nonempty integer/logical Yield forces double.
                unknown_input = true;
            }
            ValueKindFact::Numeric(_) | ValueKindFact::Logical => double_forced = true,
            ValueKindFact::Unknown => unknown_input = true,
            _ => {}
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
    let output_class = if double_forced {
        Some(NumericClass::Double)
    } else if unknown_input || (saw_single && unknown_double_geometry) {
        None
    } else if saw_single {
        Some(NumericClass::Single)
    } else {
        Some(NumericClass::Double)
    };
    let mut output = if let Some(class) = output_class {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            shape.clone(),
            if all_host {
                if class == NumericClass::Double && shape.element_count() == Some(1) {
                    StorageFact::Scalar
                } else {
                    StorageFact::Dense
                }
            } else {
                StorageFact::Unknown
            },
        )
    } else {
        // Input class or scalar geometry is unresolved at compile time.
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.shape = shape.clone();
        fact
    };
    output.residency = if all_host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    output.shape = shape;
    finish_fixed_outputs(entry, request, vec![output.clone(), output], diagnostics)
}
