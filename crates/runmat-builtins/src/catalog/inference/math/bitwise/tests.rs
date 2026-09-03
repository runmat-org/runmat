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
