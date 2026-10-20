use super::*;
use futures::executor::block_on;
use runmat_value::{ForeignRef, Value};
use std::rc::Rc;

fn context() -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

fn fixture_path() -> std::path::PathBuf {
    if cfg!(windows) {
        std::path::PathBuf::from(r"C:\sealed-fixture.mat")
    } else {
        std::path::PathBuf::from("/sealed-fixture.mat")
    }
}

fn object(runtime: Option<RuntimeContext>) -> crate::execution::RootedValue {
    block_on(crate::builtins::io::mat::matfile::matfile_builtin(
        runtime,
        Value::String(fixture_path().to_string_lossy().into_owned().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("issue MAT receiver")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn managed_matfile_clone_is_valid_but_public_fields_cannot_retarget_it() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let runtime = context();
    let issued = object(Some(runtime.clone()));
    let Value::Object(issued_object) = issued.value() else {
        panic!("MAT receiver is an object");
    };
    let clone = issued.clone();
    let Value::Object(clone_object) = clone.value() else {
        unreachable!("cloned MAT receiver")
    };
    assert!(resolve_matfile(Some(&runtime), clone_object).is_ok());

    let Value::Struct(properties) = issued_object
        .property(PROPERTIES_FIELD)
        .expect("Properties")
    else {
        panic!("Properties struct");
    };
    let forged_source = Value::String("/other.mat".into());
    let replacement = crate::execution::construct_struct_from_fields_in_context(
        &runtime,
        properties.fields.iter().map(|(name, value)| {
            (
                name,
                if name == SOURCE_FIELD {
                    &forged_source
                } else {
                    value
                },
            )
        }),
    )
    .expect("admit forged presentation");
    let forged = replacement.with_owned(|replacement| {
        let Value::Struct(replacement) = replacement else {
            unreachable!("structure constructor")
        };
        crate::execution::RootedValue::in_context(Value::Object(issued_object.clone()), &runtime)
            .expect("root forged receiver source")
            .try_map_owned(
                |value| {
                    let Value::Object(mut object) = value else {
                        unreachable!("rooted object source")
                    };
                    crate::execution::object::replacement::replace_property(
                        Some(&runtime),
                        &mut object,
                        PROPERTIES_FIELD,
                        Value::Struct(replacement),
                        |message, source| {
                            crate::context::host_allocation::host_allocation_runtime_error(
                                Some(&runtime),
                                "RunMat:matfile:TestAllocation",
                                message,
                                source,
                            )
                        },
                        matfile_error,
                    )?;
                    Ok(Value::Object(object))
                },
                |error| error.into_runtime_error(),
            )
            .expect("publish forged receiver")
    });
    let Value::Object(forged_object) = forged.value() else {
        unreachable!("forged receiver publication")
    };
    assert!(resolve_matfile(Some(&runtime), forged_object).is_err());
    assert!(resolve_matfile(Some(&runtime), issued_object).is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn revoked_captured_filesystem_keeps_typed_io_cause_on_receiver_resolution() {
    use crate::builtins::io::mat::tests::receipt::GatedOpenFs;
    use runmat_filesystem as fs;

    let _state = crate::builtins::io::mat::tests::state_guard();
    let directory = "/mat";
    let filename = "/mat/revoked.mat";
    let (provider, _release) = GatedOpenFs::new(directory, filename, b"unused");
    let provider = std::sync::Arc::new(provider);
    let _selected = fs::replace_provider(provider.clone());
    fs::set_current_dir(directory).expect("virtual cwd");
    let issued = block_on(crate::builtins::io::mat::matfile::matfile_builtin(
        None,
        Value::from(filename),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("context-free MAT receiver");
    let Value::Object(issued) = issued.value() else {
        panic!("MAT receiver must be an object")
    };
    provider.revoke_fs();
    let error = resolve_matfile(None, issued)
        .err()
        .expect("revoked owner must reject");
    assert_eq!(error.identifier(), Some("RunMat:matfile:Io"));
    let source = std::error::Error::source(&error).expect("typed filesystem cause");
    assert!(source.downcast_ref::<std::io::Error>().is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn detached_cross_session_and_ended_session_receivers_reject() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let runtime = context();
    let foreign = context();
    let issued = object(Some(runtime.clone()));
    let Value::Object(issued_object) = issued.value() else {
        panic!("MAT receiver is an object");
    };
    assert!(resolve_matfile(Some(&foreign), issued_object).is_err());

    let original = issued_object.matfile_authority().expect("sealed owner");
    let detached = ForeignRef::detached(
        original.resource_key(),
        original.type_identity.clone(),
        original.ownership,
        original.affinity,
        original.lifetime,
    );
    let synthetic = crate::execution::object::construct_object_from_fields_with_native(
        Some(&runtime),
        issued_object.class_name().display_name(),
        issued_object.properties().iter(),
        Some(crate::execution::object::ObjectNativeInitialization::MatfileAuthority(detached)),
    )
    .expect("root detached fixture");
    let Value::Object(synthetic) = synthetic.value() else {
        unreachable!("detached fixture object")
    };
    assert!(resolve_matfile(Some(&runtime), synthetic).is_err());

    drop(runtime);
    assert!(resolve_matfile(Some(&foreign), issued_object).is_err());
    assert!(resolve_matfile(None, issued_object).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn context_free_embedding_is_honestly_process_local() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let issued = object(None);
    let Value::Object(issued) = issued.value() else {
        panic!("MAT receiver is an object");
    };
    let unrelated = context();
    let resolved = resolve_matfile(Some(&unrelated), issued).expect("embedding owner");
    assert_eq!(resolved.source.path(), fixture_path());
    assert_eq!(
        issued.matfile_authority().expect("owner").lifetime,
        ForeignLifetime::Persistent
    );
}
