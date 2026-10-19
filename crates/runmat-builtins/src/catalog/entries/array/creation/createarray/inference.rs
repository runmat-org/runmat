use crate::catalog::inference::{
    argument_error, default_double_scalar, finish_fixed, literal_text, numeric_kind,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueKindFact,
};

#[derive(Clone, Copy)]
enum CreationClass {
    Numeric(NumericClass),
    Logical,
    Complex,
}

impl CreationClass {
    fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("logical") {
            Some(Self::Logical)
        } else if text.eq_ignore_ascii_case("complex") {
            Some(Self::Complex)
        } else {
            NumericClass::from_class_name(text).map(Self::Numeric)
        }
    }
}

#[derive(Clone, Copy)]
enum FillComplexity {
    Real,
    Complex,
    Unknown,
}

fn fill_complexity(request: &CallRequest, fill: Option<usize>) -> FillComplexity {
    let Some(index) = fill else {
        return FillComplexity::Real;
    };
    match request.arguments.get(index).map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(numeric)) if numeric.domain == NumericDomain::Complex => {
            FillComplexity::Complex
        }
        Some(ValueKindFact::Numeric(_) | ValueKindFact::Logical) => FillComplexity::Real,
        _ => FillComplexity::Unknown,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut dimensions = Vec::new();
    let mut prototype = None;
    let mut fill = None;
    let mut class = None;
    let mut diagnostics = Vec::new();
    let mut invalid_size_layout = false;
    let mut index = 0;
    while index < request.arguments.len() {
        let keyword = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text);
        match keyword.as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("like") => {
                if prototype.replace(index + 1).is_some() {
                    diagnostics.push(argument_error(
                        "RM-CREATEARRAY-LIKE",
                        "Like may be specified only once",
                        index,
                    ));
                }
                index += 2;
                continue;
            }
            Some("fillvalue") => {
                if fill.replace(index + 1).is_some() {
                    diagnostics.push(argument_error(
                        "RM-CREATEARRAY-FILL",
                        "FillValue may be specified only once",
                        index,
                    ));
                }
                index += 2;
                continue;
            }
            Some(text) => {
                if let Some(parsed) = CreationClass::parse(text) {
                    class = Some(parsed);
                    index += 1;
                    continue;
                }
            }
            _ => {}
        }
        if let Some(vector) = request.literals.numeric_vector_at(index) {
            if vector.len() > 1
                && request
                    .arguments
                    .get(index)
                    .and_then(|argument| argument.shape.known_dims())
                    .is_some_and(|shape| shape.len() > 1 && shape[0].is_some_and(|rows| rows > 1))
            {
                diagnostics.push(argument_error(
                    "RM-CREATEARRAY-DIMS",
                    "size vectors must be row vectors or rank-one vectors",
                    index,
                ));
                invalid_size_layout = true;
            }
            dimensions.extend(vector);
        } else {
            dimensions.push(super::super::shape_facts::scalar_dimension(request, index));
        }
        index += 1;
    }
    let mut output = prototype
        .and_then(|index| request.arguments.get(index))
        .or_else(|| fill.and_then(|index| request.arguments.get(index)))
        .cloned()
        .unwrap_or_else(default_double_scalar);
    if prototype.is_none()
        && matches!(
            output.kind,
            ValueKindFact::Character | ValueKindFact::String
        )
    {
        output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
    }
    if class.is_some() && prototype.is_some() {
        diagnostics.push(argument_error(
            "RM-CREATEARRAY-CLASS-LIKE",
            "class and Like controls cannot be combined",
            0,
        ));
    }
    // The Runtime's fill output dispatch promotes numeric classes (including
    // integer classes) on complex FillValue. Logical class stays logical,
    // whereas a logical Like prototype with complex fill becomes Double
    // complex. An unknown fill cannot prove a real output from a real Like or
    // explicit numeric class.
    let fill_complexity = fill_complexity(request, fill);
    output.kind = match (class, fill_complexity) {
        (Some(CreationClass::Logical), _) => ValueKindFact::Logical,
        (Some(CreationClass::Complex), _) => {
            numeric_kind(NumericClass::Double, NumericDomain::Complex)
        }
        (Some(CreationClass::Numeric(class)), FillComplexity::Real) => {
            numeric_kind(class, NumericDomain::Real)
        }
        (Some(CreationClass::Numeric(class)), FillComplexity::Complex) => {
            numeric_kind(class, NumericDomain::Complex)
        }
        (Some(CreationClass::Numeric(_)), FillComplexity::Unknown) => ValueKindFact::Unknown,
        (None, FillComplexity::Complex) if output.kind == ValueKindFact::Logical => {
            numeric_kind(NumericClass::Double, NumericDomain::Complex)
        }
        (None, FillComplexity::Complex) => match output.numeric() {
            Some(numeric) => numeric_kind(numeric.class, NumericDomain::Complex),
            None => ValueKindFact::Unknown,
        },
        (None, FillComplexity::Unknown)
            if !output
                .numeric()
                .is_some_and(|numeric| numeric.domain == NumericDomain::Complex) =>
        {
            ValueKindFact::Unknown
        }
        (None, _) => output.kind,
    };
    output.shape = if invalid_size_layout {
        ShapeFact::Unknown
    } else if dimensions.is_empty() {
        ShapeFact::Scalar
    } else {
        super::super::shape_facts::normalized(dimensions)
    };
    output.storage = match output.shape.element_count() {
        Some(1)
            if matches!(
                output.kind,
                ValueKindFact::Numeric(runmat_types::NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            ) && output.residency == ResidencyFact::Host =>
        {
            StorageFact::Scalar
        }
        Some(1) if output.kind == ValueKindFact::Unknown => StorageFact::Unknown,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    if prototype.is_none() && fill.is_none() {
        output.residency = ResidencyFact::Host;
    } else if matches!(output.residency, ResidencyFact::Unknown) {
        output.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, NumericFact, OutputSelection, ValueFact};

    #[test]
    fn complex_fill_promotes_real_single_like_without_copying_prototype_shape() {
        let scalar = ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real));
        let prototype = ValueFact::proven(
            numeric_kind(NumericClass::Single, NumericDomain::Real),
            ShapeFact::from(vec![Some(5), Some(7)]),
            StorageFact::Dense,
        );
        let fill = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }));
        let request = CallRequest {
            arguments: vec![
                scalar,
                ValueFact::scalar(ValueKindFact::String),
                prototype,
                ValueFact::scalar(ValueKindFact::String),
                fill,
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Number(2.0),
                LiteralValue::String("Like".into()),
                LiteralValue::Unknown,
                LiteralValue::String("FillValue".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::CREATEARRAY_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(2)])
        );
        assert_eq!(
            result.outputs[0].kind,
            numeric_kind(NumericClass::Single, NumericDomain::Complex)
        );
    }

    #[test]
    fn complex_fill_follows_runtime_class_and_like_dispatch() {
        let size = ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real));
        let complex = ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Complex));
        for (class, expected) in [
            (
                "double",
                numeric_kind(NumericClass::Double, NumericDomain::Complex),
            ),
            (
                "single",
                numeric_kind(NumericClass::Single, NumericDomain::Complex),
            ),
            (
                "int16",
                numeric_kind(NumericClass::Int16, NumericDomain::Complex),
            ),
            ("logical", ValueKindFact::Logical),
        ] {
            let request = CallRequest {
                arguments: vec![
                    size.clone(),
                    ValueFact::scalar(ValueKindFact::String),
                    ValueFact::scalar(ValueKindFact::String),
                    complex.clone(),
                ],
                literals: LiteralContext::new(vec![
                    LiteralValue::Number(2.0),
                    LiteralValue::String(class.into()),
                    LiteralValue::String("FillValue".into()),
                    LiteralValue::Unknown,
                ]),
                outputs: OutputSelection::default(),
            };
            assert_eq!(
                infer(&request, &super::super::CREATEARRAY_CATALOG_ENTRY).outputs[0].kind,
                expected,
                "{class} FillValue domain",
            );
        }
        let logical_like = CallRequest {
            arguments: vec![
                size,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::scalar(ValueKindFact::String),
                complex,
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Number(2.0),
                LiteralValue::String("Like".into()),
                LiteralValue::Unknown,
                LiteralValue::String("FillValue".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&logical_like, &super::super::CREATEARRAY_CATALOG_ENTRY).outputs[0].kind,
            numeric_kind(NumericClass::Double, NumericDomain::Complex),
        );

        let unknown_fill = CallRequest {
            arguments: vec![
                ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real)),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Number(1.0),
                LiteralValue::String("single".into()),
                LiteralValue::String("FillValue".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&unknown_fill, &super::super::CREATEARRAY_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    }

    #[test]
    fn row_size_vectors_and_empty_vectors_keep_positional_dimensions() {
        let size = ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real));
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    size.kind.clone(),
                    ShapeFact::from(vec![Some(1), Some(2)]),
                    StorageFact::Dense,
                ),
                size.clone(),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)]),
                LiteralValue::Number(4.0),
            ]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::CREATEARRAY_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3), Some(4)])
        );

        request.literals.literal_args[0] = LiteralValue::Vector(Vec::new());
        request.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(0)]);
        let result = infer(&request, &super::super::CREATEARRAY_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(4)])
        );

        request.literals.literal_args[0] =
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)]);
        request.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(1)]);
        let result = infer(&request, &super::super::CREATEARRAY_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    }
}
