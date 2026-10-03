use super::Filter2Option;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DimensionFact,
    DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILTER2-ARITY",
            "filter2 requires H and X",
            0,
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILTER2-OUTPUTS",
            "filter2 returns one result",
            1,
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        let value = &request.arguments[index];
        if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILTER2-CLASS",
                "H and X must be dense real numeric or logical arrays",
                index,
            ));
        }
    }
    let mut shape_mode = Some(Filter2Option::Same);
    for index in 2..request.arguments.len() {
        let option = &request.arguments[index];
        if !matches!(
            option.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILTER2-OPTION-TYPE",
                "filter2 options must be strings",
                index,
            ));
        }
        match request.literals.literal_string_at(index) {
            Some(value) => match Filter2Option::from_name(&value) {
                Some(mode @ (Filter2Option::Same | Filter2Option::Full | Filter2Option::Valid)) => {
                    shape_mode = Some(mode)
                }
                Some(Filter2Option::Convolution | Filter2Option::Correlation) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-FILTER2-OPTION",
                    "expected same, full, valid, conv, or corr",
                    index,
                )),
            },
            None => shape_mode = None,
        }
    }
    let kernel = request.arguments.first();
    let image = request.arguments.get(1);
    let shape = match (shape_mode, image) {
        (Some(Filter2Option::Same), Some(image)) => image.shape.clone(),
        (Some(Filter2Option::Full | Filter2Option::Valid), Some(image)) => geometry(
            &image.shape,
            kernel.map(|kernel| &kernel.shape),
            shape_mode.unwrap(),
        ),
        _ => ShapeFact::Unknown,
    };
    let class = [kernel, image]
        .into_iter()
        .flatten()
        .filter_map(|value| match value.kind {
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }) => Some(class),
            ValueKindFact::Logical => Some(NumericClass::Double),
            _ => None,
        })
        .collect::<Vec<_>>();
    let result_class = if class.contains(&NumericClass::Single) {
        Some(NumericClass::Single)
    } else if class.len() == 2 {
        Some(NumericClass::Double)
    } else {
        None
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = result_class.map_or(ValueKindFact::Unknown, |class| {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })
    });
    output.shape = shape;
    output.storage = StorageFact::Dense;
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn geometry(image: &ShapeFact, kernel: Option<&ShapeFact>, mode: Filter2Option) -> ShapeFact {
    let (Some(ShapeFact::Shaped { dims: image }), Some(ShapeFact::Shaped { dims: kernel })) =
        (Some(image), kernel)
    else {
        return ShapeFact::Unknown;
    };
    if image.len() != 2 || kernel.len() != 2 {
        return ShapeFact::Unknown;
    }
    let dims = image
        .iter()
        .zip(kernel)
        .map(|(image, kernel)| match (image, kernel) {
            (DimensionFact::Known(image), DimensionFact::Known(kernel)) if *kernel > 0 => {
                match mode {
                    Filter2Option::Full => {
                        (*image).checked_add(*kernel - 1).map(DimensionFact::Known)
                    }
                    Filter2Option::Valid => Some(DimensionFact::Known(if image >= kernel {
                        *image - *kernel + 1
                    } else {
                        0
                    })),
                    _ => unreachable!("only full/valid request geometry"),
                }
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    dims.map_or(ShapeFact::Ranked { rank: 2 }, |dims| ShapeFact::Shaped {
        dims,
    })
}
