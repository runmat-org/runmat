use super::*;
use crate::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

#[test]
fn all_five_bindings_infer_the_correct_host_object_class() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.identity.name)
            .collect::<Vec<_>>(),
        [
            "timetable",
            "array2timetable",
            "table2timetable",
            "timetable2table",
            "readtimetable"
        ]
    );
    for (entry, arguments, class) in [
        (
            &constructor::ENTRY,
            vec![],
            runmat_types::standard::TIMETABLE,
        ),
        (
            &array2timetable::ENTRY,
            vec![unknown()],
            runmat_types::standard::TIMETABLE,
        ),
        (
            &table2timetable::ENTRY,
            vec![unknown()],
            runmat_types::standard::TIMETABLE,
        ),
        (
            &timetable2table::ENTRY,
            vec![unknown()],
            runmat_types::standard::TABLE,
        ),
        (
            &readtimetable::ENTRY,
            vec![unknown()],
            runmat_types::standard::TIMETABLE,
        ),
    ] {
        let inferred = infer_catalog_call(entry, &call(arguments, 1));
        assert!(inferred.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(inferred.outputs.len(), 1);
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
        let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
            panic!("{} must infer a tabular object", entry.identity.name)
        };
        assert_eq!(object.runtime_class, Some(class.owned()));
        assert!(!object.properties_complete);
        assert_eq!(object.handle_semantics, Some(false));
    }
}

#[test]
fn impossible_inputs_are_diagnosed_without_inventing_table_schema() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    for entry in [
        &table2timetable::ENTRY,
        &timetable2table::ENTRY,
        &readtimetable::ENTRY,
    ] {
        assert!(
            !infer_catalog_call(entry, &call(vec![numeric.clone()], 1))
                .diagnostics
                .is_empty(),
            "{}",
            entry.identity.name
        );
    }
    for entry in [
        &array2timetable::ENTRY,
        &table2timetable::ENTRY,
        &timetable2table::ENTRY,
        &readtimetable::ENTRY,
    ] {
        assert!(
            !infer_catalog_call(entry, &call(vec![], 1))
                .diagnostics
                .is_empty(),
            "{}",
            entry.identity.name
        );
    }
}

#[test]
fn file_import_is_an_effectful_host_boundary() {
    let result = infer_catalog_call(&readtimetable::ENTRY, &call(vec![unknown()], 1));
    assert!(result.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(result.effects.0.contains(&EffectKind::EnvironmentRead));
    assert!(result.effects.0.contains(&EffectKind::MaySuspend));
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        readtimetable::ENTRY.contract.capabilities,
        &[CapabilityRequirement::Filesystem]
    );
}

#[test]
fn in_memory_timetable_construction_does_not_require_a_filesystem() {
    for entry in [
        &constructor::ENTRY,
        &array2timetable::ENTRY,
        &table2timetable::ENTRY,
        &timetable2table::ENTRY,
    ] {
        assert!(
            entry.contract.capabilities.is_empty(),
            "{}",
            entry.identity.name
        );
    }
}

#[test]
fn array2timetable_named_timing_signatures_describe_each_argument() {
    for (signature, option, value_type) in [
        (
            &ARRAY2TIMETABLE_DESCRIPTOR.signatures[0],
            "RowTimes",
            BuiltinParamType::Any,
        ),
        (
            &ARRAY2TIMETABLE_DESCRIPTOR.signatures[1],
            "SampleRate",
            BuiltinParamType::NumericScalar,
        ),
        (
            &ARRAY2TIMETABLE_DESCRIPTOR.signatures[2],
            "TimeStep",
            BuiltinParamType::Any,
        ),
    ] {
        assert_eq!(signature.inputs.len(), 3, "{}", signature.label);
        assert_eq!(signature.inputs[0].name, "X");
        assert_eq!(signature.inputs[1].name, option);
        assert_eq!(signature.inputs[1].ty, BuiltinParamType::StringScalar);
        assert_eq!(signature.inputs[2].ty, value_type);
    }
}
