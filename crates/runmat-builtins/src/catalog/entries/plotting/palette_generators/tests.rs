use super::ENTRIES;
use super::*;
use crate::*;
use runmat_types::*;

fn numeric(shape: ShapeFact, class: NumericClass, residency: ResidencyFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    value.residency = residency;
    value
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn every_named_palette_has_two_identity_owned_forms_and_exact_integer_capability() {
    assert_eq!(ENTRIES.len(), 19);
    for entry in ENTRIES {
        let name = entry.identity.name;
        let labels = entry
            .descriptor
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&format!("c = {name}()").as_str()), "{name}");
        assert!(
            labels.contains(&format!("c = {name}(m)").as_str()),
            "{name}"
        );
        assert_eq!(entry.integer_capabilities.len(), 1, "{name}");
        let integer = &entry.integer_capabilities[0];
        assert_eq!(integer.inputs[0].classes.len(), 8, "{name}");
        assert_eq!(integer.output_class, BuiltinIntegerOutputClassRule::Double);
        assert_eq!(integer.backend, BuiltinIntegerBackendRule::HostOnly);
        assert_eq!(entry.bindings, REQUIRED_DEFAULT_BINDING.as_slice());
    }
}

#[test]
fn palette_inference_distinguishes_dynamic_default_literal_and_invalid_controls() {
    for entry in ENTRIES {
        let default = shared::infer(&request(vec![], vec![], 1), entry);
        assert!(default.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(
            default.outputs[0].shape,
            ShapeFact::from(vec![None, Some(3)])
        );
        assert!(default.effects.0.contains(&EffectKind::EnvironmentRead));
        let exact = shared::infer(
            &request(
                vec![numeric(
                    ShapeFact::from(vec![Some(1), Some(1)]),
                    NumericClass::UInt64,
                    ResidencyFact::Host,
                )],
                vec![LiteralValue::Integer {
                    text: "7".into(),
                    class: NumericClass::UInt64,
                }],
                1,
            ),
            entry,
        );
        assert!(exact.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(
            exact.outputs[0].shape,
            ShapeFact::from(vec![Some(7), Some(3)])
        );
        assert!(!exact.effects.0.contains(&EffectKind::EnvironmentRead));
        let unknown = shared::infer(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                vec![],
                1,
            ),
            entry,
        );
        assert!(unknown.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(
            unknown.outputs[0].shape,
            ShapeFact::from(vec![None, Some(3)])
        );
        for bad in [
            numeric(
                ShapeFact::from(vec![Some(2), Some(1)]),
                NumericClass::Double,
                ResidencyFact::Host,
            ),
            numeric(
                ShapeFact::from(vec![Some(1), Some(1)]),
                NumericClass::Double,
                ResidencyFact::Device { provider: None },
            ),
        ] {
            assert!(
                !shared::infer(&request(vec![bad], vec![], 1), entry)
                    .diagnostics
                    .is_empty(),
                "{}",
                entry.identity.name
            );
        }
        assert!(
            !shared::infer(
                &request(
                    vec![numeric(
                        ShapeFact::from(vec![Some(1), Some(1)]),
                        NumericClass::Double,
                        ResidencyFact::Host,
                    )],
                    vec![LiteralValue::Number(-1.0)],
                    1,
                ),
                entry,
            )
            .diagnostics
            .is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(shared::infer(&request(vec![], vec![], 0), entry)
            .outputs
            .is_empty());
        assert!(!shared::infer(&request(vec![], vec![], 2), entry)
            .diagnostics
            .is_empty());
    }
}
