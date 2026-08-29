use crate::{
    AliasFact, ContiguityFact, DynamicReason, FactInference, InferenceDiagnostic, LayoutFact,
    MutationFact, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact, ViewFact,
};

/// Infer a value converted to one of the built-in numeric classes.
///
/// Numeric conversion preserves logical shape, sparse/dense representation,
/// complex domain, and placement. It materializes a new value, so view and
/// alias facts must not be copied from the source expression.
pub fn infer_numeric_conversion(source: &ValueFact, target: NumericClass) -> FactInference {
    let domain = match source.kind {
        ValueKindFact::Numeric(numeric) => numeric.domain,
        ValueKindFact::Logical | ValueKindFact::Character => NumericDomain::Real,
        ValueKindFact::Unknown => NumericDomain::Real,
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
    output.kind = ValueKindFact::Numeric(NumericFact {
        class: target,
        domain,
    });
    output.storage = conversion_storage(&source.shape, source.storage);
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

fn conversion_storage(shape: &ShapeFact, source: StorageFact) -> StorageFact {
    if shape.element_count() == Some(1) {
        return StorageFact::Scalar;
    }
    match source {
        StorageFact::Sparse => StorageFact::Sparse,
        StorageFact::Scalar | StorageFact::Dense => StorageFact::Dense,
        StorageFact::Unknown | StorageFact::Opaque => StorageFact::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CertaintyFact, DimensionFact, ResidencyFact};

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

        let output = infer_numeric_conversion(&source, NumericClass::UInt64).fact;
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
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
    fn numeric_conversion_retains_target_class_when_input_shape_is_dynamic() {
        let source = ValueFact::unknown(DynamicReason::RuntimeValue);
        let output = infer_numeric_conversion(&source, NumericClass::Int16).fact;
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int16,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(output.shape, ShapeFact::Unknown);
        assert_eq!(output.storage, StorageFact::Unknown);
    }
}
