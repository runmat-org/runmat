use super::*;
use crate::{
    catalog::inference::infer_catalog_call, BuiltinOutputMode, BuiltinParamArity,
    BuiltinPortability,
};
use crate::{BuiltinAcceleratorPolicy, BuiltinLinkPolicy, BuiltinPurity, ForeignAdapterFamily};
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

#[test]
fn native_identities_are_canonical_and_ordered() {
    let names: std::collections::BTreeSet<_> =
        ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(
        names,
        std::collections::BTreeSet::from([
            "calllib",
            "clibgen.buildInterface",
            "libfunctions",
            "libisloaded",
            "libpointer",
            "libstruct",
            "loadlibrary",
            "setdatatype",
            "unloadlibrary",
        ])
    );
    for entry in ENTRIES {
        assert_eq!(entry.placement.portability, BuiltinPortability::NativeOnly);
        assert!(entry
            .contract
            .effects
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
        assert!(entry
            .contract
            .capabilities
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
        assert!(entry
            .contract
            .capabilities
            .contains(&CapabilityRequirement::NativeCode));
        assert!(entry
            .contract
            .capabilities
            .contains(&CapabilityRequirement::ForeignRuntime));
    }
}

#[test]
fn loaded_state_and_declared_functions_have_useful_shapes() {
    let loaded = infer_catalog_call(&LIBISLOADED_CATALOG_ENTRY, &request(vec![text()], 1));
    assert_eq!(loaded.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(loaded.outputs[0].shape, ShapeFact::Scalar);

    let names = infer_catalog_call(&LIBFUNCTIONS_CATALOG_ENTRY, &request(vec![text()], 1));
    let ValueKindFact::Cell(cell) = &names.outputs[0].kind else {
        panic!("expected cell of declared symbol names");
    };
    assert_eq!(cell.element.kind, ValueKindFact::String);
    assert_eq!(names.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
}

#[test]
fn pointer_facts_keep_owned_origin_thread_session_lifetime() {
    for entry in [&LIBPOINTER_CATALOG_ENTRY, &LIBSTRUCT_CATALOG_ENTRY] {
        let result = infer_catalog_call(entry, &request(vec![text()], 1));
        let ValueKindFact::Foreign(reference) = &result.outputs[0].kind else {
            panic!("expected native foreign handle");
        };
        assert_eq!(reference.ownership, ForeignOwnershipFact::Owned);
        assert_eq!(reference.affinity, ForeignAffinityFact::OriginThread);
        assert_eq!(reference.lifetime, ForeignLifetimeFact::Session);
        assert_eq!(result.outputs[0].alias, AliasFact::Identity);
    }
}

#[test]
fn call_outputs_remain_dynamic_and_statement_forms_return_nothing() {
    let call = infer_catalog_call(&CALLLIB_CATALOG_ENTRY, &request(vec![text(), text()], 3));
    assert_eq!(call.outputs.len(), 3);
    assert!(call
        .outputs
        .iter()
        .all(|fact| fact.kind == ValueKindFact::Unknown));
    assert!(call.effects.0.contains(&EffectKind::HostCallback));
    assert!(call.effects.0.contains(&EffectKind::Unknown));
    for entry in [&SETDATATYPE_CATALOG_ENTRY, &UNLOADLIBRARY_CATALOG_ENTRY] {
        let result = infer_catalog_call(entry, &request(vec![text()], 0));
        assert!(result.outputs.is_empty());
    }
}

#[test]
fn load_options_use_one_typed_case_insensitive_vocabulary() {
    assert_eq!(
        NativeBuildOption::parse("LIBRARIES"),
        Some(NativeBuildOption::Libraries)
    );
    assert_eq!(
        NativeBuildOption::parse("interfaceName"),
        Some(NativeBuildOption::InterfaceName)
    );
    assert_eq!(
        NativeBuildOption::parse("IncludeDirectories"),
        Some(NativeBuildOption::IncludePath)
    );
    assert_eq!(NativeBuildOption::parse("compilerflags"), None);
    for spelling in ["Alias", "alias", "IncludePath", "includePATH", "AddHeader"] {
        assert!(NativeLoadOption::parse(spelling).is_some());
    }
    assert_eq!(NativeLoadOption::parse("compilerflags"), None);
    let mut invalid = request(vec![text(), text(), text(), text()], 1);
    invalid.literals = LiteralContext::new(vec![
        LiteralValue::String("fixture".into()),
        LiteralValue::String("fixture.h".into()),
        LiteralValue::String("CompilerFlags".into()),
        LiteralValue::String("-O2".into()),
    ]);
    let result = infer_catalog_call(&LOADLIBRARY_CATALOG_ENTRY, &invalid);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-NATIVE-FFI-OPTION"));
}

#[test]
fn qualified_interface_preparation_has_one_canonical_process_link_and_text_contract() {
    let entry = crate::builtin_catalog_entry_by_name("clibgen.buildInterface").unwrap();
    assert_eq!(
        entry.identity,
        CLIBGEN_BUILD_INTERFACE_CATALOG_ENTRY.identity
    );
    assert_eq!(
        entry.link.policy,
        BuiltinLinkPolicy::ForeignRuntime(ForeignAdapterFamily::NativeFfi)
    );
    assert_eq!(
        entry.link.execution_stack,
        ExecutionStackRequirement::Process
    );
    assert_eq!(entry.contract.purity, BuiltinPurity::Impure);
    assert_eq!(
        entry.placement.accelerator,
        BuiltinAcceleratorPolicy::Forbidden
    );
    let mut valid = request(vec![text(), text(), text()], 1);
    valid.literals = LiteralContext::new(vec![
        LiteralValue::String("fixture.h".into()),
        LiteralValue::String("Libraries".into()),
        LiteralValue::String("fixture".into()),
    ]);
    let result = infer_catalog_call(entry, &valid);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::String);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::NativeCode));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::ForeignRuntime));
    assert!(result.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(result.effects.0.contains(&EffectKind::FilesystemWrite));
    valid.literals = LiteralContext::new(vec![
        LiteralValue::String("fixture.h".into()),
        LiteralValue::String("InterfaceName".into()),
        LiteralValue::String("fixture".into()),
    ]);
    assert!(infer_catalog_call(entry, &valid)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-NATIVE-FFI-LIBRARY"));
}

#[test]
fn retained_scalar_value_to_text_forms_are_not_false_errors() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let numeric_alias = infer_catalog_call(&LIBISLOADED_CATALOG_ENTRY, &request(vec![numeric], 1));
    assert!(!numeric_alias
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-NATIVE-FFI-TEXT"));

    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(text()),
            elements: vec![text()],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Opaque,
    );
    let invalid = infer_catalog_call(&LIBISLOADED_CATALOG_ENTRY, &request(vec![cell], 1));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-NATIVE-FFI-TEXT"));

    let paged_char = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(3), Some(1)]),
        StorageFact::Opaque,
    );
    let invalid_page =
        infer_catalog_call(&LIBISLOADED_CATALOG_ENTRY, &request(vec![paged_char], 1));
    assert!(invalid_page
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-NATIVE-FFI-TEXT"));
}

#[test]
fn identity_descriptors_match_fixed_output_facts_and_dynamic_call_abi() {
    let cases = [
        (&LOADLIBRARY_CATALOG_ENTRY, 2),
        (&UNLOADLIBRARY_CATALOG_ENTRY, 0),
        (&LIBISLOADED_CATALOG_ENTRY, 1),
        (&LIBFUNCTIONS_CATALOG_ENTRY, 1),
        (&LIBPOINTER_CATALOG_ENTRY, 1),
        (&LIBSTRUCT_CATALOG_ENTRY, 1),
        (&SETDATATYPE_CATALOG_ENTRY, 0),
    ];
    for (entry, maximum) in cases {
        let descriptor_maximum = entry
            .descriptor
            .signatures
            .iter()
            .map(|signature| signature.outputs.len())
            .max()
            .unwrap_or(0);
        assert_eq!(descriptor_maximum, maximum, "{}", entry.identity.name);
        assert!(entry.descriptor.signatures.iter().all(|signature| signature
            .outputs
            .iter()
            .all(|output| output.arity == BuiltinParamArity::Required)));
    }
    assert_eq!(
        LOADLIBRARY_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    for requested in [0, 1, 2] {
        let result = infer_catalog_call(
            &LOADLIBRARY_CATALOG_ENTRY,
            &request(vec![text()], requested),
        );
        assert_eq!(result.outputs.len(), requested);
    }
    assert!(CALLLIB_DESCRIPTOR.signatures[0]
        .outputs
        .iter()
        .any(|output| output.arity == BuiltinParamArity::Variadic));
    assert_eq!(
        infer_catalog_call(&CALLLIB_CATALOG_ENTRY, &request(vec![text(), text()], 4))
            .outputs
            .len(),
        4
    );
}
