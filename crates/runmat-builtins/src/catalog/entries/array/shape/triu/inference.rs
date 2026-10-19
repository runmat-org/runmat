use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-TRIU-ARITY",
            "triu requires one or two inputs",
            0,
        ));
    }
    let Some(source) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let supported = source.storage != StorageFact::Sparse
        && !matches!(source.residency, ResidencyFact::Remote { .. })
        && matches!(
            source.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
        );
    if !supported {
        diagnostics.push(argument_error(
            "RM-TRIU-INPUT",
            "triu requires a dense numeric, logical, character, complex, or resident array",
            0,
        ));
    }
    if let Some(offset) = request.arguments.get(1) {
        let valid = !offset.shape.is_definitely_nonscalar()
            && offset.storage != StorageFact::Sparse
            && !matches!(offset.residency, ResidencyFact::Remote { .. })
            && matches!(
                offset.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            )
            || (!offset.shape.is_definitely_nonscalar()
                && offset.storage == StorageFact::Scalar
                && !matches!(offset.residency, ResidencyFact::Remote { .. })
                && offset.kind == ValueKindFact::Logical);
        let literal_invalid = request
            .literals
            .literal_args
            .get(1)
            .is_some_and(|value| !offset_literal_supported(value));
        if !valid || literal_invalid {
            diagnostics.push(argument_error(
                "RM-TRIU-OFFSET",
                "triu diagonal offset must be a finite integer-valued real scalar in range",
                1,
            ));
        }
    }
    let mut output = source.clone();
    if source.storage == StorageFact::Scalar && source.kind == ValueKindFact::Logical {
        output.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
    }
    let proven_host_value = source.residency == ResidencyFact::Host
        && source.kind != ValueKindFact::Unknown
        && source.storage != StorageFact::Unknown;
    output.alias = if proven_host_value {
        AliasFact::Unique
    } else {
        AliasFact::Unknown
    };
    output.view = if proven_host_value {
        ViewFact::Materialized
    } else {
        ViewFact::Unknown
    };
    output.contiguity = if proven_host_value {
        ContiguityFact::Contiguous
    } else {
        ContiguityFact::Unknown
    };
    output.layout = if proven_host_value {
        if source.kind == ValueKindFact::Character {
            source.layout
        } else {
            LayoutFact::ColumnMajor
        }
    } else {
        LayoutFact::Unknown
    };
    if matches!(source.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
        output.storage = StorageFact::Unknown;
        output.alias = AliasFact::Unknown;
        output.view = ViewFact::Unknown;
    }
    if !diagnostics.is_empty() {
        output.shape = ShapeFact::Unknown;
        output.certainty = CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation);
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn offset_literal_supported(value: &LiteralValue) -> bool {
    if matches!(value, LiteralValue::Unknown | LiteralValue::Bool(_)) {
        return true;
    }
    if let LiteralValue::Vector(values) = value {
        return values.len() == 1 && offset_literal_supported(&values[0]);
    }
    if let LiteralValue::Matrix(rows) = value {
        return rows.len() == 1 && rows[0].len() == 1 && offset_literal_supported(&rows[0][0]);
    }
    if let Some(integer) = value.integer_value_i128() {
        return isize::try_from(integer).is_ok();
    }
    LiteralContext::numeric_from_literal(value).is_some_and(|number| {
        number.is_finite()
            && (number.round() - number).abs() <= f64::EPSILON
            && number >= isize::MIN as f64
            && number < isize::MAX as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn numeric(class: NumericClass, domain: NumericDomain, shape: &[usize]) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact { class, domain }),
            ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
            StorageFact::Dense,
        )
    }
    fn call(
        arguments: Vec<ValueFact>,
        literals: Vec<LiteralValue>,
        outputs: usize,
    ) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
            },
            &super::super::TRIU_CATALOG_ENTRY,
        )
    }
    #[test]
    fn preserves_nd_shape_class_and_marks_device_fallback_uncertain() {
        let source = numeric(NumericClass::UInt64, NumericDomain::Real, &[2, 3, 4]);
        let result = call(vec![source.clone()], vec![LiteralValue::Unknown], 1);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, source.kind);
        assert_eq!(result.outputs[0].shape, source.shape);
        let mut resident = numeric(NumericClass::Single, NumericDomain::Real, &[2, 3]);
        resident.residency = ResidencyFact::Device {
            provider: Some("provider".into()),
        };
        resident.storage = StorageFact::Opaque;
        let result = call(vec![resident], vec![LiteralValue::Unknown], 1);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
        assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
        let unknown = call(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            vec![LiteralValue::Unknown],
            1,
        );
        assert_eq!(unknown.outputs[0].alias, AliasFact::Unknown);
        assert_eq!(unknown.outputs[0].view, ViewFact::Unknown);
    }
    #[test]
    fn rejects_bad_arity_output_count_sparse_input_and_invalid_offsets() {
        let source = numeric(NumericClass::Double, NumericDomain::Real, &[2, 2]);
        assert!(!call(vec![], vec![], 1).diagnostics.is_empty());
        assert!(!call(vec![source.clone()], vec![LiteralValue::Unknown], 2)
            .diagnostics
            .is_empty());
        let mut sparse = source.clone();
        sparse.storage = StorageFact::Sparse;
        assert!(!call(vec![sparse], vec![LiteralValue::Unknown], 1)
            .diagnostics
            .is_empty());
        let mut remote = source.clone();
        remote.residency = ResidencyFact::Remote { pool: None };
        assert!(!call(vec![remote], vec![LiteralValue::Unknown], 1)
            .diagnostics
            .is_empty());
        let offset = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        assert!(!call(
            vec![source.clone(), offset.clone()],
            vec![LiteralValue::Unknown, LiteralValue::Number(f64::NAN)],
            1
        )
        .diagnostics
        .is_empty());
        assert!(call(
            vec![source, offset],
            vec![LiteralValue::Unknown, LiteralValue::Number(-2.0)],
            0
        )
        .diagnostics
        .is_empty());
        let source = numeric(NumericClass::Double, NumericDomain::Real, &[2, 2]);
        let dense_logical = ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
        );
        assert!(!call(
            vec![source.clone(), dense_logical],
            vec![LiteralValue::Unknown, LiteralValue::Unknown],
            1,
        )
        .diagnostics
        .is_empty());
        assert!(call(
            vec![source, ValueFact::scalar(ValueKindFact::Logical)],
            vec![LiteralValue::Unknown, LiteralValue::Bool(false)],
            1,
        )
        .diagnostics
        .is_empty());
    }
}
