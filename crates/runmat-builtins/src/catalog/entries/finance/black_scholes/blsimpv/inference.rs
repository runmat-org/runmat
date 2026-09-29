use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    if args.len() < 5 || !(args.len() - 5).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BLSIMPV-ARITY",
            "blsimpv expects five inputs and name-value pairs",
            args.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-BLSIMPV-OUTPUTS",
            "blsimpv returns one output",
            0,
        ));
    }
    let mut shape = ShapeFact::Scalar;
    for (index, value) in args.iter().take(5).enumerate() {
        if !matches!(
            value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || value
            .numeric()
            .is_some_and(|numeric| numeric.domain == NumericDomain::Complex)
            || value.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLSIMPV-INPUT",
                "pricing inputs must be real numeric or logical arrays",
                index,
            ));
        }
        if matches!(value.residency, ResidencyFact::Remote { .. }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLSIMPV-PLACEMENT",
                "distributed inputs are not supported",
                index,
            ));
        }
        shape = match broadcast_shape(&shape, &value.shape) {
            Ok(next) => next,
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-BLSIMPV-SHAPE",
                    "pricing inputs cannot be scalar-expanded together",
                    index,
                ));
                ShapeFact::Unknown
            }
        };
    }
    for index in (5..args.len()).step_by(2) {
        let name = &args[index];
        if !matches!(
            name.kind,
            ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLSIMPV-OPTION",
                "option name must be scalar text",
                index,
            ));
        }
        if let Some(value) = args.get(index + 1) {
            if matches!(value.residency, ResidencyFact::Remote { .. }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-BLSIMPV-PLACEMENT",
                    "distributed options are not supported",
                    index + 1,
                ));
            }
            match request
                .literals
                .literal_string_at(index)
                .as_deref()
                .map(str::to_ascii_lowercase)
            {
                Some(option) if option == "yield" => {
                    if value.shape.element_count() != Some(0) {
                        shape = match broadcast_shape(&shape, &value.shape) {
                            Ok(next) => next,
                            Err(_) => {
                                diagnostics.push(argument_error(
                                    "RM-CATALOG-BLSIMPV-SHAPE",
                                    "Yield cannot be scalar-expanded with pricing inputs",
                                    index + 1,
                                ));
                                ShapeFact::Unknown
                            }
                        };
                    }
                }
                Some(option) if option == "limit" || option == "tolerance" => {
                    if value.shape.element_count().is_some_and(|count| count > 1) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-BLSIMPV-OPTION",
                            "Limit and Tolerance must be scalar",
                            index + 1,
                        ));
                    }
                }
                Some(option) if option == "class" => {
                    if !matches!(value.kind, ValueKindFact::Character)
                        && value.shape.element_count().is_some_and(|count| count > 1)
                        && !value.shape.is_proven_equivalent(&shape)
                        && value.shape.known_dims().is_some()
                        && shape.known_dims().is_some()
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-BLSIMPV-CLASS",
                            "nonscalar Class must match pricing output shape",
                            index + 1,
                        ));
                    }
                }
                Some(option) if option == "method" => {}
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-BLSIMPV-OPTION",
                    "unsupported option name",
                    index,
                )),
                None => shape = ShapeFact::Unknown,
            }
        }
    }
    let output = ValueFact::proven(
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
    );
    finish_fixed(entry, request, output, diagnostics)
}
