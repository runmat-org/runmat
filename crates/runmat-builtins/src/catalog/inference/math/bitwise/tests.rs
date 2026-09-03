use super::*;
use crate::SWAPBYTES_CATALOG_ENTRY;
use runmat_types::{
    AliasFact, CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn request(argument: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn request_many(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn numeric(class: NumericClass, shape: impl Into<ShapeFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape.into(),
        StorageFact::Dense,
    )
}

#[test]
fn swapbytes_preserves_real_numeric_facts_and_materializes_on_the_host() {
    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    input.residency = ResidencyFact::Device {
        provider: Some("wgpu".into()),
    };
    input.alias = AliasFact::Shared;
    let result = infer(
        crate::BitwiseInferenceRule::SwapBytes,
        &request(input),
        &SWAPBYTES_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[0].alias, AliasFact::Unique);
}

#[test]
fn swapbytes_rejects_complex_sparse_and_nonnumeric_inputs() {
    for input in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int16,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Sparse,
        ),
        ValueFact::scalar(ValueKindFact::Logical),
    ] {
        let result = infer(
            crate::BitwiseInferenceRule::SwapBytes,
            &request(input),
            &SWAPBYTES_CATALOG_ENTRY,
        );
        assert!(!result.diagnostics.is_empty());
    }
}

#[test]
fn binary_bitwise_preserves_integer_class_and_broadcast_shape() {
    let result = infer(
        crate::BitwiseInferenceRule::Binary(crate::BinaryBitwiseOperator::And),
        &request_many(vec![
            numeric(NumericClass::UInt16, vec![Some(2), Some(1)]),
            numeric(NumericClass::UInt16, vec![Some(1), Some(3)]),
        ]),
        &crate::BITAND_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
}

#[test]
fn direct_bitwise_rules_retain_the_data_class() {
    let value = numeric(NumericClass::Int32, vec![Some(1), Some(4)]);
    let control = numeric(NumericClass::UInt8, ShapeFact::Scalar);
    for (rule, entry, arguments) in [
        (
            crate::BitwiseInferenceRule::Complement,
            &crate::BITCMP_CATALOG_ENTRY,
            vec![value.clone()],
        ),
        (
            crate::BitwiseInferenceRule::Get,
            &crate::BITGET_CATALOG_ENTRY,
            vec![value.clone(), control.clone()],
        ),
        (
            crate::BitwiseInferenceRule::Set,
            &crate::BITSET_CATALOG_ENTRY,
            vec![value.clone(), control.clone(), control.clone()],
        ),
        (
            crate::BitwiseInferenceRule::Shift,
            &crate::BITSHIFT_CATALOG_ENTRY,
            vec![value.clone(), control.clone()],
        ),
    ] {
        let result = infer(rule, &request_many(arguments), entry);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, value.kind);
        assert_eq!(result.outputs[0].shape, value.shape);
    }
}
