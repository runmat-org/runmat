use super::{argument_error, finish_fixed, numeric_kind, preserved_binary_residency};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape, AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason,
    LayoutFact, MutationFact, NumericClass, NumericDomain, ShapeFact, StorageFact, ValueFact,
    ValueKindFact, ViewFact,
};

pub(super) fn infer_gamrnd(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GAMRND-ARITY",
            "gamrnd requires shape and scale parameters",
            request.arguments.len(),
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }

    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        if matches!(argument.storage, StorageFact::Sparse)
            || !matches!(
                argument.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAMRND-PARAMETER",
                "gamrnd parameters must be dense real numeric values",
                index,
            ));
        } else if matches!(
            argument.kind,
            ValueKindFact::Numeric(runmat_types::NumericFact {
                domain: NumericDomain::Complex,
                ..
            })
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAMRND-PARAMETER",
                "gamrnd parameters must be real",
                index,
            ));
        }
    }

    for (index, argument) in request.arguments.iter().enumerate().skip(2) {
        if matches!(argument.storage, StorageFact::Sparse)
            || !matches!(
                argument.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAMRND-SIZE",
                "gamrnd size controls must be dense real numeric values",
                index,
            ));
        }
    }

    let parameters = &request.arguments[..2];
    let output_class = if parameters.iter().any(|argument| {
        matches!(
            argument.kind,
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: NumericClass::Single,
                ..
            })
        )
    }) {
        Some(NumericClass::Single)
    } else if parameters
        .iter()
        .all(|argument| matches!(argument.kind, ValueKindFact::Numeric(_)))
    {
        Some(NumericClass::Double)
    } else {
        None
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let Some(output_class) = output_class {
        output.kind = numeric_kind(output_class, NumericDomain::Real);
    }
    output.shape = gamrnd_shape(request, &mut diagnostics);
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    output.residency = preserved_binary_residency(
        &request.arguments[0].residency,
        &request.arguments[1].residency,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn gamrnd_shape(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ShapeFact {
    if request.arguments.len() == 2 {
        return match broadcast_shape(&request.arguments[0].shape, &request.arguments[1].shape) {
            Ok(shape) => shape,
            Err(diagnostic) => {
                diagnostics.push(diagnostic);
                ShapeFact::Unknown
            }
        };
    }

    let mut dimensions = request.literals.numeric_dims_from(2);
    if request.arguments.len() == 3 {
        if let Some(vector) = request.literals.numeric_vector_at(2) {
            dimensions = vector;
        } else if dimensions.first().is_some_and(Option::is_some) {
            dimensions.push(dimensions[0]);
        } else {
            return ShapeFact::Ranked { rank: 2 };
        }
    }
    if dimensions.is_empty() {
        return ShapeFact::Unknown;
    }
    while dimensions.len() > 2 && dimensions.last() == Some(&Some(1)) {
        dimensions.pop();
    }
    ShapeFact::from(dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
    use runmat_types::{
        DimensionFact, LiteralContext, LiteralValue, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact,
    };

    fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
        let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }));
        fact.shape = shape;
        fact.storage = StorageFact::Dense;
        fact
    }

    #[test]
    fn gamrnd_tracks_parameter_class_shape_size_and_residency() {
        let entry = builtin_catalog_entry_by_name("gamrnd").expect("gamrnd catalog entry");
        let mut shape = numeric(
            NumericClass::Single,
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)],
            },
        );
        shape.residency = ResidencyFact::Device {
            provider: Some("test-provider".into()),
        };
        let scale = numeric(NumericClass::Double, ShapeFact::Scalar);
        let request = CallRequest {
            arguments: vec![shape, scale],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let inferred = infer_catalog_call(entry, &request);
        assert!(inferred.diagnostics.is_empty());
        let output = &inferred.outputs[0];
        assert!(matches!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real
            })
        ));
        assert_eq!(output.shape.known_dims(), Some(vec![Some(1), Some(3)]));
        assert!(matches!(output.residency, ResidencyFact::Device { .. }));

        let request = CallRequest {
            arguments: vec![
                numeric(NumericClass::Double, ShapeFact::Scalar),
                numeric(NumericClass::Double, ShapeFact::Scalar),
                numeric(NumericClass::Double, ShapeFact::Scalar),
            ],
            outputs: OutputSelection::new(RequestedOutputCount::One),
            literals: LiteralContext::new(vec![
                LiteralValue::Number(2.0),
                LiteralValue::Number(3.0),
                LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(4.0)]),
            ]),
        };
        let inferred = infer_catalog_call(entry, &request);
        assert_eq!(
            inferred.outputs[0].shape.known_dims(),
            Some(vec![Some(2), Some(4)])
        );

        let request = CallRequest {
            arguments: vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                numeric(NumericClass::Double, ShapeFact::Scalar),
            ],
            outputs: OutputSelection::new(RequestedOutputCount::One),
            literals: LiteralContext::default(),
        };
        let inferred = infer_catalog_call(entry, &request);
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    }
}
