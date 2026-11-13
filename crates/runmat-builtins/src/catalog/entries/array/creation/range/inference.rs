use crate::catalog::inference::{finish_fixed, literal_text, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            Vec::new(),
        );
    };
    // The host implementation widens exact integer differences to double. A
    // real Single input remains Single; resident results require the provider
    // result's actual physical class, which is not proven by this static fact.
    let kind = match input.numeric() {
        Some(numeric) if numeric.domain == NumericDomain::Real => numeric_kind(
            if numeric.class == NumericClass::Single {
                NumericClass::Single
            } else {
                NumericClass::Double
            },
            NumericDomain::Real,
        ),
        _ if matches!(input.kind, ValueKindFact::Logical) => {
            numeric_kind(NumericClass::Double, NumericDomain::Real)
        }
        _ => ValueKindFact::Unknown,
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = if matches!(input.residency, ResidencyFact::Host) {
        kind
    } else {
        ValueKindFact::Unknown
    };
    output.shape = if matches!(input.residency, ResidencyFact::Host) {
        host_result_shape(request, &input.shape)
    } else {
        ShapeFact::Unknown
    };
    let scalar = output.shape.element_count() == Some(1);
    output.storage = if scalar
        && matches!(
            output.kind,
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: NumericClass::Double,
                ..
            })
        ) {
        StorageFact::Scalar
    } else if matches!(output.kind, ValueKindFact::Numeric(_))
        && matches!(input.residency, ResidencyFact::Host)
    {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    if matches!(input.residency, ResidencyFact::Host) {
        output.residency = ResidencyFact::Host;
    }
    finish_fixed(entry, request, output, Vec::new())
}

fn host_result_shape(request: &CallRequest, input: &ShapeFact) -> ShapeFact {
    let Some(mut dims) = input.known_dims() else {
        return ShapeFact::Unknown;
    };
    let mut selected: Option<Vec<Option<usize>>> = None;
    let mut all = false;
    for index in 1..request.arguments.len() {
        if let Some(text) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            match text.to_ascii_lowercase().as_str() {
                "omitnan" | "includenan" => continue,
                "all" => {
                    all = true;
                    continue;
                }
                _ => return ShapeFact::Unknown,
            }
        }
        if selected.is_some() || all {
            return ShapeFact::Unknown;
        }
        selected = request.literals.numeric_vector_at(index).or_else(|| {
            request
                .literals
                .numeric_dims_from(index)
                .first()
                .copied()
                .map(|dim| vec![dim])
        });
        if selected.is_none() {
            return ShapeFact::Unknown;
        }
    }
    if all {
        dims.fill(Some(1));
    } else {
        let selected = selected.unwrap_or_default();
        if selected.is_empty() {
            // The default is the first nonsingleton dimension. An unknown
            // extent before it makes that choice runtime-dependent.
            if dims.iter().any(Option::is_none) {
                return ShapeFact::Unknown;
            }
            let dimension = dims.iter().position(|dim| *dim != Some(1)).unwrap_or(0);
            dims[dimension] = Some(1);
        } else {
            for dim in selected {
                let Some(dim) = dim.filter(|dim| *dim > 0) else {
                    return ShapeFact::Unknown;
                };
                if dim <= dims.len() {
                    dims[dim - 1] = Some(1);
                }
            }
        }
    }
    if dims.iter().all(|dim| *dim == Some(1)) {
        ShapeFact::Scalar
    } else {
        ShapeFact::from(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, NumericFact, OutputSelection};

    #[test]
    fn single_matrix_default_and_explicit_axes_retain_class_and_shape() {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let mut request = CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::default(),
        };
        let output = &infer(&request, &super::super::RANGE_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.shape, ShapeFact::from(vec![Some(1), Some(3)]));
        assert_eq!(
            output.kind,
            numeric_kind(NumericClass::Single, NumericDomain::Real)
        );

        request.arguments.push(ValueFact::scalar(numeric_kind(
            NumericClass::Double,
            NumericDomain::Real,
        )));
        request
            .literals
            .literal_args
            .push(LiteralValue::Number(2.0));
        let output = &infer(&request, &super::super::RANGE_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(1)]));
    }
}
