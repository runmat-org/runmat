//! Facts shared by constant-filled numeric and logical constructors.
//! Size argument geometry is never used as output dimension contents.
use crate::catalog::inference::{argument_error, finish_fixed, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
pub(super) enum FillClass {
    Logical,
    Floating,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: FillClass,
) -> CallInference {
    let logical = matches!(class, FillClass::Logical);
    let mut output = ValueFact::scalar(if logical {
        ValueKindFact::Logical
    } else {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    });
    let mut diagnostics = Vec::new();
    let mut dimensions = Vec::new();
    let mut vector = false;
    let mut unknown_dimensions = false;
    let mut unknown_options = false;
    let mut invalid_dimensions = false;
    let mut class_option = false;
    let mut prototype = None;
    let mut explicit_device = false;
    let mut index = 0;
    while index < request.arguments.len() {
        let literal = request.literals.literal_args.get(index);
        if let Some(text) = literal.and_then(literal_text) {
            match text.to_ascii_lowercase().as_str() {
                "like" => {
                    if prototype.is_some() || index + 1 >= request.arguments.len() {
                        diagnostics.push(argument_error(
                            "RM-CONSTRUCTOR-LIKE",
                            "like requires exactly one prototype",
                            index,
                        ));
                    } else {
                        prototype = request.arguments.get(index + 1);
                    }
                    index += 2;
                    continue;
                }
                "single" | "double" if !logical => {
                    if class_option || explicit_device {
                        diagnostics.push(argument_error(
                            "RM-CONSTRUCTOR-CLASS-CONFLICT",
                            "class and gpuArray specifiers cannot be repeated or combined",
                            index,
                        ));
                    }
                    class_option = true;
                    output.kind = ValueKindFact::Numeric(NumericFact {
                        class: if text.eq_ignore_ascii_case("single") {
                            NumericClass::Single
                        } else {
                            NumericClass::Double
                        },
                        domain: NumericDomain::Real,
                    });
                }
                "gpuarray" if !logical => {
                    if class_option || explicit_device {
                        diagnostics.push(argument_error(
                            "RM-CONSTRUCTOR-CLASS-CONFLICT",
                            "class and gpuArray specifiers cannot be repeated or combined",
                            index,
                        ));
                    }
                    explicit_device = true;
                }
                "logical" if logical => {}
                _ => diagnostics.push(argument_error(
                    "RM-CONSTRUCTOR-OPTION",
                    "unrecognized constructor option",
                    index,
                )),
            }
        } else {
            match literal {
                Some(LiteralValue::Vector(values)) => {
                    if vector || !dimensions.is_empty() {
                        diagnostics.push(argument_error(
                            "RM-CONSTRUCTOR-DIMS",
                            "a size vector cannot be combined with other dimensions",
                            index,
                        ));
                    }
                    vector = true;
                    for value in values {
                        invalid_dimensions |=
                            !matches!(value, LiteralValue::Unknown) && dimension(value).is_none();
                        unknown_dimensions |= matches!(value, LiteralValue::Unknown);
                        dimensions.push(dimension(value));
                    }
                }
                Some(LiteralValue::Empty) => vector = true,
                Some(
                    value @ (LiteralValue::Number(_)
                    | LiteralValue::Real { .. }
                    | LiteralValue::Integer { .. }),
                ) => {
                    if vector {
                        diagnostics.push(argument_error(
                            "RM-CONSTRUCTOR-DIMS",
                            "a size vector cannot be combined with other dimensions",
                            index,
                        ));
                    }
                    invalid_dimensions |= dimension(value).is_none();
                    dimensions.push(dimension(value));
                }
                _ => {
                    // Runtime may interpret a non-vector logical argument as an
                    // extension prototype. Numeric vector contents remain unknown.
                    unknown_dimensions = true;
                    if matches!(
                        request.arguments[index].kind,
                        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                    ) || (logical && !definite_size_control(&request.arguments[index]))
                    {
                        // Logical constructors also admit implicit prototypes.
                        // An unknown matrix can therefore select sparse/device
                        // output; it is not just an unknown size vector.
                        unknown_options = true;
                        output = ValueFact::unknown(DynamicReason::RuntimeValue);
                        if logical {
                            output.kind = ValueKindFact::Logical;
                        }
                    }
                }
            }
        }
        index += 1;
    }
    if invalid_dimensions {
        diagnostics.push(argument_error(
            "RM-CONSTRUCTOR-DIMS",
            "dimensions must be finite integers within the platform range",
            0,
        ));
    }
    output.shape = if unknown_dimensions {
        ShapeFact::Unknown
    } else {
        if dimensions.is_empty() {
            dimensions = if vector {
                vec![Some(0), Some(0)]
            } else {
                vec![Some(1), Some(1)]
            };
        } else if dimensions.len() == 1 {
            dimensions.push(dimensions[0]);
        }
        while dimensions.len() > 2 && dimensions.last() == Some(&Some(1)) {
            dimensions.pop();
        }
        ShapeFact::from(dimensions)
    };
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else if output.shape.element_count().is_none() {
        // Unknown dimensions can still evaluate to 1-by-1 and return a scalar.
        StorageFact::Unknown
    } else {
        StorageFact::Dense
    };
    if let Some(prototype) = prototype {
        if class_option || explicit_device {
            diagnostics.push(argument_error(
                "RM-CONSTRUCTOR-CLASS-CONFLICT",
                "like cannot be combined with class or gpuArray specifiers",
                0,
            ));
        }
        if !logical {
            output.kind = prototype.kind.clone();
            if matches!(prototype.kind, ValueKindFact::Numeric(NumericFact { class, .. }) if !matches!(class, NumericClass::Double | NumericClass::Single))
            {
                diagnostics.push(argument_error(
                    "RM-CONSTRUCTOR-INTEGER-LIKE",
                    "integer prototypes cannot represent nonfinite values",
                    0,
                ));
            }
            if !matches!(
                prototype.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
            ) {
                diagnostics.push(argument_error(
                    "RM-CONSTRUCTOR-LIKE-CLASS",
                    "nonfinite constructors require a single or double prototype",
                    0,
                ));
                output.kind = ValueKindFact::Unknown;
            }
        } else if !matches!(
            prototype.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CONSTRUCTOR-LIKE-CLASS",
                "logical constructors require a numeric or logical prototype",
                0,
            ));
        }
        output.residency = prototype.residency.clone();
        if matches!(
            prototype.storage,
            StorageFact::Sparse | StorageFact::Unknown
        ) {
            output.storage = prototype.storage;
        }
    }
    if explicit_device {
        // The unqualified device form uses the selected provider's precision,
        // not a statically guaranteed double result.
        output.kind = ValueKindFact::Unknown;
        output.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        output.residency = ResidencyFact::Device { provider: None };
        output.storage = StorageFact::Dense;
    }
    if matches!(output.residency, ResidencyFact::Device { .. }) {
        output.storage = StorageFact::Dense;
    }
    if matches!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        })
    ) && prototype
        .is_none_or(|value| !matches!(value.storage, StorageFact::Sparse | StorageFact::Unknown))
    {
        // RunMat preserves single scalars in typed tensor storage.
        output.storage = StorageFact::Dense;
    }
    if prototype.is_some() && matches!(output.residency, ResidencyFact::Unknown) {
        output.storage = StorageFact::Unknown;
    }
    if unknown_options {
        output.storage = StorageFact::Unknown;
        output.residency = ResidencyFact::Unknown;
        if !logical {
            output.kind = ValueKindFact::Unknown;
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn definite_size_control(argument: &ValueFact) -> bool {
    if !matches!(
        argument.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
    ) || matches!(
        argument.storage,
        StorageFact::Sparse | StorageFact::Unknown | StorageFact::Opaque
    ) {
        return false;
    }
    argument.is_scalar()
        || argument
            .shape
            .known_dims()
            .is_some_and(|dims| dims.first() == Some(&Some(1)))
}

fn dimension(value: &LiteralValue) -> Option<usize> {
    if let LiteralValue::Integer { text, .. } = value {
        if text.starts_with('-') {
            return text.parse::<i128>().ok().map(|_| 0);
        }
        return text.parse::<usize>().ok();
    }
    let value = runmat_types::LiteralContext::numeric_from_literal(value)?;
    if !value.is_finite() || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return None;
    }
    Some(if value <= 0.0 { 0 } else { value as usize })
}

#[cfg(test)]
#[path = "constructor_facts/consumer_cutover_tests.rs"]
mod consumer_cutover_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dimension_controls_are_exact_and_clamp_negative_values() {
        assert_eq!(
            dimension(&LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64
            }),
            usize::try_from(9007199254740993u64).ok()
        );
        assert_eq!(dimension(&LiteralValue::Number(-2.0)), Some(0));
        assert_eq!(dimension(&LiteralValue::Number(1.25)), None);
        assert_eq!(dimension(&LiteralValue::Number(f64::INFINITY)), None);
    }
}
