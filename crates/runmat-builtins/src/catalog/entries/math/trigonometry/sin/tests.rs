use crate::*;
use runmat_types::EffectKind;

#[test]
fn trigonometric_like_dynamic_keywords_and_unknown_prototypes_do_not_invent_real_results() {
    use runmat_types::*;
    for entry in [
        &SIN_CATALOG_ENTRY,
        &COS_CATALOG_ENTRY,
        &TAN_CATALOG_ENTRY,
        &ATAN_CATALOG_ENTRY,
    ] {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let mut complex = input.clone();
        complex.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        });
        complex.residency = ResidencyFact::Host;
        let keyword = ValueFact::unknown(DynamicReason::RuntimeValue);
        let mut request = CallRequest {
            arguments: vec![input.clone(), keyword, complex],
            literals: LiteralContext::new(vec![LiteralValue::Unknown; 3]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = crate::catalog::inference::infer_catalog_call(entry, &request);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex
            })
        );
        assert_eq!(result.outputs[0].shape, input.shape);
        request.arguments[2] = ValueFact::unknown(DynamicReason::RuntimeValue);
        let result = crate::catalog::inference::infer_catalog_call(entry, &request);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].shape, input.shape);
        request.arguments[2] = input;
        request.arguments[2].storage = StorageFact::Sparse;
        let result = crate::catalog::inference::infer_catalog_call(entry, &request);
        assert!(!result.diagnostics.is_empty());
    }
}

#[test]
fn admitted_trigonometric_float_facts_keep_class_shape_but_not_input_placement() {
    use runmat_types::*;
    for entry in [
        &SIN_CATALOG_ENTRY,
        &COS_CATALOG_ENTRY,
        &TAN_CATALOG_ENTRY,
        &ATAN_CATALOG_ENTRY,
        &SINH_CATALOG_ENTRY,
        &COSH_CATALOG_ENTRY,
        &ASINH_CATALOG_ENTRY,
        &TANH_CATALOG_ENTRY,
    ] {
        assert!(validate_builtin_preparation(entry).is_empty());
        let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
            panic!("admitted unary form")
        };
        assert_eq!(forms[0].data_arguments, &[0]);
        assert_eq!(forms[0].operation, BuiltinPreparationOperation::Unary);
        for class in [NumericClass::Single, NumericClass::Double] {
            let mut input = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(0), Some(2), Some(3)]),
                StorageFact::Dense,
            );
            input.residency = ResidencyFact::Host;
            let request = CallRequest {
                arguments: vec![input.clone()],
                literals: LiteralContext::new(vec![LiteralValue::Unknown]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            };
            let result = crate::catalog::inference::infer_catalog_call(entry, &request);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].kind, input.kind);
            assert_eq!(result.outputs[0].shape, input.shape);
            assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
        }
    }
}

#[test]
fn trigonometric_provider_routes_declare_suspension_and_fixed_output_errors() {
    for entry in [
        &ACOS_CATALOG_ENTRY,
        &ACOSH_CATALOG_ENTRY,
        &ASIN_CATALOG_ENTRY,
        &ASINH_CATALOG_ENTRY,
        &ATAN_CATALOG_ENTRY,
        &ATAN2_CATALOG_ENTRY,
        &ATANH_CATALOG_ENTRY,
        &COS_CATALOG_ENTRY,
        &COSD_CATALOG_ENTRY,
        &COSH_CATALOG_ENTRY,
        &COSPI_CATALOG_ENTRY,
        &SIN_CATALOG_ENTRY,
        &SIND_CATALOG_ENTRY,
        &SINH_CATALOG_ENTRY,
        &SINPI_CATALOG_ENTRY,
        &TAN_CATALOG_ENTRY,
        &TAND_CATALOG_ENTRY,
        &TANH_CATALOG_ENTRY,
        &DEG2RAD_CATALOG_ENTRY,
        &RAD2DEG_CATALOG_ENTRY,
    ] {
        // sin is reviewed as argument-local numerical suspension (signal generators rely on
        // it); the rest of the family still declares ordinary suspension.
        let expected = if entry.identity.name == "sin" {
            BuiltinAsyncBehavior::MaySuspendLocally
        } else {
            BuiltinAsyncBehavior::MaySuspend
        };
        assert_eq!(
            entry.contract.async_behavior, expected,
            "{}",
            entry.identity.name
        );
        assert!(
            entry.contract.effects.contains(&EffectKind::MaySuspend),
            "{}",
            entry.identity.name
        );
        let identifier = format!("RunMat:{}:TooManyOutputs", entry.identity.name);
        assert!(
            entry
                .descriptor
                .errors
                .iter()
                .any(|error| error.identifier == Some(identifier.as_str())),
            "{}",
            entry.identity.name
        );
    }
}
