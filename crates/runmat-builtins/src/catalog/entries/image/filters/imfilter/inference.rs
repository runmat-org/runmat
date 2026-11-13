use super::ImfilterOption;
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
            "RM-CATALOG-IMFILTER-ARITY",
            "imfilter requires A and H",
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
            "RM-CATALOG-IMFILTER-OUTPUTS",
            "imfilter returns one result",
            1,
        ));
    }
    let image = request.arguments.first();
    let kernel = request.arguments.get(1);
    let output_kind = image.and_then(|value| match value.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            if matches!(class, NumericClass::Int64 | NumericClass::UInt64) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMFILTER-CLASS",
                    "A does not support 64-bit integer images",
                    0,
                ));
                None
            } else {
                Some(ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }))
            }
        }
        ValueKindFact::Logical => Some(ValueKindFact::Logical),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMFILTER-CLASS",
                "A must be real numeric or logical",
                0,
            ));
            None
        }
    });
    if image.is_some_and(|value| matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMFILTER-STORAGE",
            "A must be dense",
            0,
        ));
    }
    if let Some(kernel) = kernel {
        if !matches!(
            kernel.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            }) | ValueKindFact::Unknown
        ) || matches!(kernel.storage, StorageFact::Sparse | StorageFact::Opaque)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMFILTER-KERNEL",
                "H must be a real double scalar, vector, or matrix",
                1,
            ));
        }
        match &kernel.shape {
            ShapeFact::Shaped { dims }
                if dims.len() > 2
                    || dims
                        .iter()
                        .any(|dim| matches!(dim, DimensionFact::Known(0))) =>
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMFILTER-KERNEL-SHAPE",
                    "H must be nonempty and at most two-dimensional",
                    1,
                ));
            }
            ShapeFact::Ranked { rank } if *rank > 2 => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMFILTER-KERNEL-SHAPE",
                    "H must be at most two-dimensional",
                    1,
                ));
            }
            _ => {}
        }
    }
    if image.is_some_and(|value| matches!(&value.shape, ShapeFact::Shaped { dims } if dims.iter().any(|dim| matches!(dim, DimensionFact::Known(0))))) {
        diagnostics.push(argument_error("RM-CATALOG-IMFILTER-IMAGE-SHAPE", "A must not have zero-length dimensions", 0));
    }
    let mut shape_mode = Some(ImfilterOption::Same);
    for index in 2..request.arguments.len() {
        let option = &request.arguments[index];
        match request.literals.literal_string_at(index) {
            Some(name) => match ImfilterOption::from_name(&name) {
                Some(
                    mode @ (ImfilterOption::Same | ImfilterOption::Full | ImfilterOption::Valid),
                ) => shape_mode = Some(mode),
                Some(_) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-IMFILTER-OPTION",
                    "unknown imfilter option",
                    index,
                )),
            },
            None => match option.kind {
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
                | ValueKindFact::Logical => {
                    let known_non_scalar = match &option.shape {
                        ShapeFact::Shaped { dims }
                            if dims
                                .iter()
                                .all(|dim| matches!(dim, DimensionFact::Known(_))) =>
                        {
                            dims.iter().try_fold(1usize, |count, dim| match dim {
                                DimensionFact::Known(value) => count.checked_mul(*value),
                                _ => None,
                            }) != Some(1)
                        }
                        ShapeFact::Ranked { rank } => *rank > 2,
                        _ => false,
                    };
                    if known_non_scalar {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-IMFILTER-PAD",
                            "padding value must be scalar",
                            index,
                        ));
                    }
                }
                ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character => {
                    shape_mode = None
                }
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-IMFILTER-OPTION-TYPE",
                    "option must be a string or real numeric scalar",
                    index,
                )),
            },
        }
    }
    let shape = match (shape_mode, image) {
        (Some(ImfilterOption::Same), Some(image)) => image.shape.clone(),
        (Some(mode @ (ImfilterOption::Full | ImfilterOption::Valid)), Some(image)) => {
            geometry(&image.shape, kernel.map(|value| &value.shape), mode)
        }
        _ => ShapeFact::Unknown,
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = output_kind.unwrap_or(ValueKindFact::Unknown);
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

fn geometry(image: &ShapeFact, kernel: Option<&ShapeFact>, mode: ImfilterOption) -> ShapeFact {
    let (ShapeFact::Shaped { dims: image }, Some(ShapeFact::Shaped { dims: kernel })) =
        (image, kernel)
    else {
        return ShapeFact::Unknown;
    };
    if image.len() < 2 || kernel.len() > 2 || kernel.is_empty() {
        return ShapeFact::Unknown;
    }
    let mut dims = image.clone();
    for axis in 0..kernel.len() {
        let (DimensionFact::Known(image_dim), DimensionFact::Known(kernel_dim)) =
            (&image[axis], &kernel[axis])
        else {
            return ShapeFact::Ranked { rank: image.len() };
        };
        if *kernel_dim == 0 {
            return ShapeFact::Unknown;
        }
        let computed = match mode {
            ImfilterOption::Full => (*image_dim).checked_add(*kernel_dim - 1),
            ImfilterOption::Valid => Some(if image_dim >= kernel_dim {
                *image_dim - *kernel_dim + 1
            } else {
                0
            }),
            _ => unreachable!("only full/valid request geometry"),
        };
        let Some(computed) = computed else {
            return ShapeFact::Unknown;
        };
        dims[axis] = DimensionFact::Known(computed);
    }
    ShapeFact::Shaped { dims }
}
