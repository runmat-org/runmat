use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::*;

#[test]
fn like_residency_and_storage_distinguish_complex_host_and_resident_prototypes() {
    for name in ["double", "single"] {
        for residency in [
            ResidencyFact::Host,
            ResidencyFact::Device { provider: None },
            ResidencyFact::Unknown,
        ] {
            let mut prototype = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            }));
            prototype.residency = residency.clone();
            let inferred = infer_catalog_call(
                builtin_catalog_entry_by_name(name).unwrap(),
                &CallRequest {
                    arguments: vec![
                        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Double,
                            domain: NumericDomain::Real,
                        })),
                        ValueFact::scalar(ValueKindFact::String),
                        prototype,
                    ],
                    literals: LiteralContext::new(vec![
                        LiteralValue::Unknown,
                        LiteralValue::Keyword("like".into()),
                        LiteralValue::Unknown,
                    ]),
                    outputs: OutputSelection::new(RequestedOutputCount::One),
                },
            );
            if residency == ResidencyFact::Host {
                assert!(!inferred.diagnostics.is_empty());
            } else {
                assert!(inferred.diagnostics.is_empty());
                assert_eq!(
                    inferred.outputs[0].storage,
                    if matches!(residency, ResidencyFact::Device { .. }) {
                        StorageFact::Opaque
                    } else {
                        StorageFact::Unknown
                    }
                );
            }
        }
    }
}
