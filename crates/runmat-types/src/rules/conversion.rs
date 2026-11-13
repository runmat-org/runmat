use crate::{
    AliasFact, ContiguityFact, DynamicReason, FactInference, InferenceDiagnostic, LayoutFact,
    MutationFact, NumericClass, NumericDomain, NumericFact, StorageFact, ValueFact, ValueKindFact,
    ViewFact,
};

/// Infer a value converted to one of the built-in numeric classes.
///
/// Numeric conversion preserves logical shape, sparse/dense representation,
/// complex domain, and placement. It materializes a new value, so view and
/// alias facts must not be copied from the source expression.
pub fn infer_numeric_conversion(source: &ValueFact, target: NumericClass) -> FactInference {
    let domain = match source.kind {
        ValueKindFact::Numeric(numeric) => Some(numeric.domain),
        ValueKindFact::Logical | ValueKindFact::Character | ValueKindFact::Symbolic => {
            Some(NumericDomain::Real)
        }
        ValueKindFact::String if target == NumericClass::Double => Some(NumericDomain::Real),
        ValueKindFact::Unknown => None,
        _ => {
            return FactInference {
                fact: ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
                diagnostics: vec![InferenceDiagnostic::error(
                    "RM-TYPE-NUMERIC-CONVERSION",
                    format!(
                        "conversion to {} requires a numeric, logical, or character input",
                        target.class_name()
                    ),
                )],
            };
        }
    };

    let mut output = source.clone();
    // NumericFact couples class and domain. Until the input domain is known,
    // retaining the target class would incorrectly assert real or complex.
    output.kind = domain.map_or(ValueKindFact::Unknown, |domain| {
        ValueKindFact::Numeric(NumericFact {
            class: target,
            domain,
        })
    });
    if domain.is_none() && !matches!(output.certainty, crate::CertaintyFact::Dynamic(_)) {
        output.certainty = crate::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output.storage = conversion_storage(source, domain, target);
    output.layout = match output.storage {
        StorageFact::Scalar | StorageFact::Dense | StorageFact::Sparse => LayoutFact::ColumnMajor,
        StorageFact::Unknown | StorageFact::Opaque => LayoutFact::Unknown,
    };
    output.contiguity = match output.storage {
        StorageFact::Scalar | StorageFact::Dense => ContiguityFact::Contiguous,
        StorageFact::Sparse | StorageFact::Unknown | StorageFact::Opaque => ContiguityFact::Unknown,
    };
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    FactInference::exact(output)
}

fn conversion_storage(
    source: &ValueFact,
    domain: Option<NumericDomain>,
    target: NumericClass,
) -> StorageFact {
    if source.storage == StorageFact::Sparse {
        return StorageFact::Sparse;
    }
    // Opaque device handles are not host scalar carriers, even at one element.
    if source.storage == StorageFact::Opaque {
        return if target.integer_class().is_some()
            && matches!(source.residency, crate::ResidencyFact::Device { .. })
        {
            StorageFact::Opaque
        } else {
            StorageFact::Unknown
        };
    }
    if matches!(source.kind, ValueKindFact::Symbolic) && source.storage == StorageFact::Dense {
        return StorageFact::Dense;
    }
    // Floating tensor conversions retain their array carrier; single has no
    // scalar Value representation at all. Keep these separate from the integer
    // domain's deliberate real singleton collapse.
    if target == NumericClass::Single && !matches!(source.kind, ValueKindFact::Unknown) {
        return StorageFact::Dense;
    }
    if target == NumericClass::Double
        && source.storage == StorageFact::Dense
        && matches!(source.kind, ValueKindFact::Numeric(_))
    {
        return StorageFact::Dense;
    }
    if source.shape.element_count() == Some(1) {
        if domain.is_none() {
            return StorageFact::Unknown;
        }
        if domain == Some(NumericDomain::Complex) && target.integer_class().is_some() {
            return StorageFact::Dense;
        }
        return StorageFact::Scalar;
    }
    match source.storage {
        StorageFact::Sparse => StorageFact::Sparse,
        StorageFact::Scalar | StorageFact::Dense => StorageFact::Dense,
        StorageFact::Unknown | StorageFact::Opaque => StorageFact::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CertaintyFact, DimensionFact, ResidencyFact, ShapeFact};

    #[test]
    fn string_conversion_is_supported_only_for_double_target() {
        let source = ValueFact::scalar(ValueKindFact::String);
        let inferred = infer_numeric_conversion(&source, NumericClass::Double);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.fact.numeric().unwrap().domain, NumericDomain::Real);
        assert_eq!(inferred.fact.storage, StorageFact::Scalar);
        assert!(!infer_numeric_conversion(&source, NumericClass::Int32)
            .diagnostics
            .is_empty());
        assert!(!infer_numeric_conversion(&source, NumericClass::Single)
            .diagnostics
            .is_empty());
    }

    #[test]
    fn numeric_conversion_preserves_shape_domain_sparse_storage_and_residency() {
        let mut source = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            }),
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(3), DimensionFact::Known(4)],
            },
            StorageFact::Sparse,
        );
        source.residency = ResidencyFact::Device {
            provider: Some("test".into()),
        };

        let output = infer_numeric_conversion(&source, NumericClass::Double).fact;
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(output.shape, source.shape);
        assert_eq!(output.storage, StorageFact::Sparse);
        assert_eq!(output.residency, source.residency);
        assert_eq!(output.view, ViewFact::Materialized);
        assert_eq!(output.alias, AliasFact::Unique);
        assert_eq!(output.certainty, CertaintyFact::Proven);
    }

    #[test]
    fn numeric_conversion_does_not_invent_a_real_domain_for_unknown_input() {
        let source = ValueFact::unknown(DynamicReason::RuntimeValue);
        let output = infer_numeric_conversion(&source, NumericClass::Int16).fact;
        assert_eq!(output.kind, ValueKindFact::Unknown);
        assert_eq!(output.shape, ShapeFact::Unknown);
        assert_eq!(output.storage, StorageFact::Unknown);
    }

    #[test]
    fn numeric_conversion_types_successful_symbolic_constant_results() {
        let source = ValueFact::proven(
            ValueKindFact::Symbolic,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let output = infer_numeric_conversion(&source, NumericClass::UInt16);

        assert!(output.diagnostics.is_empty());
        assert_eq!(
            output.fact.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt16,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(output.fact.shape, source.shape);
        assert_eq!(output.fact.storage, StorageFact::Dense);
    }

    #[test]
    fn numeric_conversion_preserves_scalar_representation_distinctions() {
        for (domain, storage, expected) in [
            (NumericDomain::Real, StorageFact::Dense, StorageFact::Scalar),
            (
                NumericDomain::Complex,
                StorageFact::Dense,
                StorageFact::Dense,
            ),
            (
                NumericDomain::Real,
                StorageFact::Sparse,
                StorageFact::Sparse,
            ),
        ] {
            let source = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain,
                }),
                ShapeFact::from(vec![Some(1), Some(1)]),
                storage,
            );
            let output = infer_numeric_conversion(&source, NumericClass::Int8).fact;
            assert_eq!(output.storage, expected);
            assert_eq!(output.shape, source.shape);
            assert_eq!(output.numeric().unwrap().domain, domain);
        }
    }

    #[test]
    fn unknown_domain_retains_independent_facts_and_dynamic_reason() {
        let mut source = ValueFact::unknown(DynamicReason::ConflictingControlFlow);
        source.shape = ShapeFact::from(vec![Some(2), Some(3)]);
        source.storage = StorageFact::Dense;
        source.residency = ResidencyFact::Host;
        let output = infer_numeric_conversion(&source, NumericClass::UInt64).fact;
        assert_eq!(output.kind, ValueKindFact::Unknown);
        assert_eq!(output.shape, source.shape);
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.residency, source.residency);
        assert_eq!(output.certainty, source.certainty);
        source.shape = ShapeFact::Scalar;
        assert_eq!(
            infer_numeric_conversion(&source, NumericClass::UInt64)
                .fact
                .storage,
            StorageFact::Unknown
        );
    }
}
