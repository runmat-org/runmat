//! One linked runtime library is needed to prove the catalog-to-execution join.
use futures::executor::block_on;
use runmat_builtins::infer_catalog_call;
use runmat_builtins::{
    builtin_catalog_entry_by_name, builtin_function_by_name,
    with_builtin_implementation_provenance_by_name, BuiltinImplementationAuthority,
};
use runmat_runtime::builtin::{
    runtime_builtin_bindings_by_name, validate_runtime_builtin_registry,
};
use runmat_runtime::{call_builtin_async_with_outputs, value_fact::value_fact};
use runmat_types::{CallRequest, LiteralContext, LiteralValue, OutputSelection};
use runmat_value::{Tensor, Value};
use std::rc::Rc;

#[test]
fn eight_fft_forms_have_exact_default_bindings_without_legacy_registration() {
    for name in [
        "fft",
        "fft2",
        "fftn",
        "ifft",
        "ifft2",
        "ifftn",
        "fftshift",
        "ifftshift",
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("FFT catalog entry");
        assert_eq!(entry.bindings.len(), 1, "{name}: catalog cardinality");
        let identity = entry.binding_identity(&entry.bindings[0]);
        assert_eq!(identity.variant, "default");
        let compiled = runtime_builtin_bindings_by_name(name);
        assert_eq!(compiled.len(), 1, "{name}: compiled cardinality");
        assert_eq!(compiled[0].identity, identity, "{name}: exact binding join");
        assert!(
            builtin_function_by_name(name).is_none(),
            "{name}: legacy registration"
        );
        let owners = with_builtin_implementation_provenance_by_name(name, |all| all.to_vec());
        assert_eq!(owners.len(), 1, "{name}: unique provenance");
        assert_eq!(
            owners[0].authority,
            BuiltinImplementationAuthority::CanonicalBinding
        );
        assert_eq!(owners[0].binding_variant, Some("default"));
    }
    let registry_errors = validate_runtime_builtin_registry();
    assert!(
        registry_errors.is_empty(),
        "FFT registration: {registry_errors:#?}"
    );
}

#[test]
fn selected_fft_form_facts_match_actual_runtime_class_and_geometry() {
    let context = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    context.set_runmat_extensions_enabled(true);
    let matrix =
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap());
    let size = Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap());
    let cube =
        Value::Tensor(Tensor::new((1..=24).map(f64::from).collect(), vec![2, 3, 4]).unwrap());
    let short_size = Value::Tensor(Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap());
    for (name, args, literals) in [
        (
            "fft",
            vec![matrix.clone(), Value::Num(5.0)],
            vec![LiteralValue::Unknown, LiteralValue::Number(5.0)],
        ),
        (
            "fft2",
            vec![matrix.clone(), size.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(4.0), LiteralValue::Number(5.0)]),
            ],
        ),
        (
            "ifftn",
            vec![matrix.clone(), size, Value::from("symmetric")],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(4.0), LiteralValue::Number(5.0)]),
                LiteralValue::String("symmetric".into()),
            ],
        ),
        (
            "fftn",
            vec![cube.clone(), short_size.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(5.0), LiteralValue::Number(6.0)]),
            ],
        ),
        (
            "ifftn",
            vec![cube, short_size, Value::from("symmetric")],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(5.0), LiteralValue::Number(6.0)]),
                LiteralValue::String("symmetric".into()),
            ],
        ),
        (
            "fftshift",
            vec![matrix.clone()],
            vec![LiteralValue::Unknown],
        ),
        ("ifftshift", vec![matrix], vec![LiteralValue::Unknown]),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical FFT entry");
        let inferred = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: args.iter().map(value_fact).collect(),
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::default(),
            },
        );
        assert!(inferred.diagnostics.is_empty(), "{name}: {inferred:?}");
        let actual =
            block_on(context.scope(call_builtin_async_with_outputs(name, &args, 1))).unwrap();
        let actual = value_fact(&actual);
        assert_eq!(inferred.outputs[0].kind, actual.kind, "{name}");
        assert!(
            inferred.outputs[0]
                .shape
                .is_proven_equivalent(&actual.shape),
            "{name}: inferred {:?}, actual {:?}",
            inferred.outputs[0].shape,
            actual.shape
        );
    }
}
