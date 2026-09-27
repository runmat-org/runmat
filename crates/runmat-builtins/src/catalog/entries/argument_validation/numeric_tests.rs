use crate::*;
use runmat_types::*;
fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn numeric_validator_forms_admit_unknown_contents_but_not_public_outputs() {
    for entry in [
        &MUST_BE_FILE_CATALOG_ENTRY,
        &MUST_BE_FOLDER_CATALOG_ENTRY,
        &MUST_BE_FINITE_CATALOG_ENTRY,
        &MUST_BE_INTEGER_CATALOG_ENTRY,
        &MUST_BE_NON_NAN_CATALOG_ENTRY,
        &MUST_BE_NONZERO_CATALOG_ENTRY,
        &MUST_BE_NONMISSING_CATALOG_ENTRY,
        &MUST_BE_REAL_CATALOG_ENTRY,
        &MUST_BE_POSITIVE_CATALOG_ENTRY,
        &MUST_BE_NEGATIVE_CATALOG_ENTRY,
        &MUST_BE_NONNEGATIVE_CATALOG_ENTRY,
        &MUST_BE_NONPOSITIVE_CATALOG_ENTRY,
        &MUST_BE_GREATER_THAN_CATALOG_ENTRY,
        &MUST_BE_GREATER_THAN_OR_EQUAL_CATALOG_ENTRY,
        &MUST_BE_LESS_THAN_CATALOG_ENTRY,
        &MUST_BE_LESS_THAN_OR_EQUAL_CATALOG_ENTRY,
        &MUST_BE_IN_RANGE_CATALOG_ENTRY,
        &MUST_BE_MEMBER_CATALOG_ENTRY,
    ] {
        let inputs = entry.descriptor.signatures[0]
            .inputs
            .iter()
            .filter(|p| p.arity == BuiltinParamArity::Required)
            .count();
        let args = vec![ValueFact::unknown(DynamicReason::RuntimeValue); inputs];
        let inferred = infer_catalog_call(entry, &request(args.clone(), 0));
        assert!(
            inferred.diagnostics.is_empty(),
            "{} {:?}",
            entry.identity.name,
            inferred.diagnostics
        );
        assert!(inferred.outputs.is_empty());
        assert!(inferred.effects.0.contains(&EffectKind::MaySuspend));
        assert!(!infer_catalog_call(entry, &request(args, 1))
            .diagnostics
            .is_empty());
        assert!(!infer_catalog_call(entry, &request(vec![], 0))
            .diagnostics
            .is_empty());
    }
}
#[test]
fn literal_predicate_failures_and_range_controls_are_not_ignored() {
    for (entry, number) in [
        (&MUST_BE_FINITE_CATALOG_ENTRY, f64::INFINITY),
        (&MUST_BE_INTEGER_CATALOG_ENTRY, 0.5),
        (&MUST_BE_NON_NAN_CATALOG_ENTRY, f64::NAN),
        (&MUST_BE_NONMISSING_CATALOG_ENTRY, f64::NAN),
        (&MUST_BE_NONZERO_CATALOG_ENTRY, 0.0),
        (&MUST_BE_POSITIVE_CATALOG_ENTRY, -1.0),
        (&MUST_BE_NEGATIVE_CATALOG_ENTRY, 1.0),
        (&MUST_BE_NONNEGATIVE_CATALOG_ENTRY, -1.0),
        (&MUST_BE_NONPOSITIVE_CATALOG_ENTRY, 1.0),
    ] {
        let mut call = request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 0);
        call.literals.literal_args = vec![LiteralValue::Number(number)];
        assert!(
            !infer_catalog_call(entry, &call).diagnostics.is_empty(),
            "{}",
            entry.identity.name
        );
    }
    for (flags, rejected) in [
        (vec!["openleft"], false),
        (vec!["inclusive", "exclusive"], false),
        (vec!["openleft", "exclusive"], true),
        (vec!["bad"], true),
    ] {
        let mut call = request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue); 3 + flags.len()],
            0,
        );
        call.literals.literal_args = vec![LiteralValue::Unknown; 3];
        call.literals.literal_args.extend(
            flags
                .into_iter()
                .map(|flag| LiteralValue::String(flag.into())),
        );
        assert_eq!(
            !infer_catalog_call(&MUST_BE_IN_RANGE_CATALOG_ENTRY, &call)
                .diagnostics
                .is_empty(),
            rejected
        );
    }
}
#[test]
fn filesystem_validation_declares_environment_read_and_suspension() {
    for entry in [&MUST_BE_FILE_CATALOG_ENTRY, &MUST_BE_FOLDER_CATALOG_ENTRY] {
        let inferred = infer_catalog_call(
            entry,
            &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 0),
        );
        for effect in [
            EffectKind::EnvironmentRead,
            EffectKind::FilesystemRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ] {
            assert!(inferred.effects.0.contains(&effect));
        }
    }
}

#[test]
fn ordered_literal_validators_reject_nan_without_changing_signed_zero_boundaries() {
    for (entry, accepts_zero) in [
        (&MUST_BE_POSITIVE_CATALOG_ENTRY, false),
        (&MUST_BE_NEGATIVE_CATALOG_ENTRY, false),
        (&MUST_BE_NONNEGATIVE_CATALOG_ENTRY, true),
        (&MUST_BE_NONPOSITIVE_CATALOG_ENTRY, true),
    ] {
        for (value, accepted) in [(f64::NAN, false), (0.0, accepts_zero), (-0.0, accepts_zero)] {
            let mut call = request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 0);
            call.literals.literal_args = vec![LiteralValue::Number(value)];
            assert_eq!(
                infer_catalog_call(entry, &call).diagnostics.is_empty(),
                accepted,
                "{} {value:?}",
                entry.identity.name
            );
        }
    }
    for (entry, accepts_equal) in [
        (&MUST_BE_GREATER_THAN_CATALOG_ENTRY, false),
        (&MUST_BE_GREATER_THAN_OR_EQUAL_CATALOG_ENTRY, true),
        (&MUST_BE_LESS_THAN_CATALOG_ENTRY, false),
        (&MUST_BE_LESS_THAN_OR_EQUAL_CATALOG_ENTRY, true),
    ] {
        for (left, right, accepted) in [
            (f64::NAN, 1.0, false),
            (1.0, f64::NAN, false),
            (f64::NAN, f64::NAN, false),
            (-0.0, 0.0, accepts_equal),
            (f64::INFINITY, f64::INFINITY, accepts_equal),
        ] {
            let mut call = request(vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2], 0);
            call.literals.literal_args =
                vec![LiteralValue::Number(left), LiteralValue::Number(right)];
            assert_eq!(
                infer_catalog_call(entry, &call).diagnostics.is_empty(),
                accepted,
                "{} {left:?} {right:?}",
                entry.identity.name
            );
        }
    }
}
