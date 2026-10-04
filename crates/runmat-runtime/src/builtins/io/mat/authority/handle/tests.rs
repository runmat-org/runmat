use super::*;
use futures::executor::block_on;
use runmat_value::{ForeignRef, ObjectInstance, Value};
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

fn object(runtime: Option<RuntimeContext>) -> ObjectInstance {
    let value = block_on(crate::builtins::io::mat::matfile::matfile_builtin(
        runtime,
        Value::String(fixture_path().to_string_lossy().into_owned()),
        vec![],
    ))
    .expect("issue MAT receiver");
    let Value::Object(object) = value else {
        panic!("MAT receiver is an object");
    };
    object
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn managed_matfile_clone_is_valid_but_public_fields_cannot_retarget_it() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let runtime = context();
    let issued = object(Some(runtime.clone()));
    let clone = issued.clone();
    assert!(resolve_matfile(Some(&runtime), &clone).is_ok());

    let mut forged = issued.clone();
    let Value::Struct(properties) = forged
        .properties
        .get_mut(PROPERTIES_FIELD)
        .expect("Properties")
    else {
        panic!("Properties struct");
    };
    properties
        .fields
        .insert(SOURCE_FIELD.into(), Value::String("/other.mat".into()));
    assert!(resolve_matfile(Some(&runtime), &forged).is_err());
    assert!(resolve_matfile(Some(&runtime), &issued).is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn revoked_captured_filesystem_keeps_typed_io_cause_on_receiver_resolution() {
    use crate::builtins::io::mat::tests::receipt::GatedOpenFs;
    use runmat_filesystem as fs;

    let _state = crate::builtins::io::mat::tests::state_guard();
    let directory = if cfg!(windows) { r"C:\mat" } else { "/mat" };
    let filename = if cfg!(windows) {
        r"C:\mat\revoked.mat"
    } else {
        "/mat/revoked.mat"
    };
    let (provider, _release) = GatedOpenFs::new(directory, filename, b"unused");
    let provider = std::sync::Arc::new(provider);
    let _selected = fs::replace_provider(provider.clone());
    fs::set_current_dir(directory).expect("virtual cwd");
    let issued = block_on(crate::builtins::io::mat::matfile::matfile_builtin(
        None,
        Value::from(filename),
        vec![],
    ))
    .expect("context-free MAT receiver");
    let Value::Object(issued) = issued else {
        panic!("MAT receiver must be an object")
    };
    provider.revoke_fs();
    let error = resolve_matfile(None, &issued)
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
    assert!(resolve_matfile(Some(&foreign), &issued).is_err());

    let original = issued.matfile_authority().expect("sealed owner");
    let detached = ForeignRef::detached(
        original.resource_key(),
        original.type_identity.clone(),
        original.ownership,
        original.affinity,
        original.lifetime,
    );
    let synthetic = issued.clone().with_matfile_authority(detached);
    assert!(resolve_matfile(Some(&runtime), &synthetic).is_err());

    drop(runtime);
    assert!(resolve_matfile(Some(&foreign), &issued).is_err());
    assert!(resolve_matfile(None, &issued).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn context_free_embedding_is_honestly_process_local() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let issued = object(None);
    let unrelated = context();
    let resolved = resolve_matfile(Some(&unrelated), &issued).expect("embedding owner");
    assert_eq!(resolved.source.path(), fixture_path());
    assert_eq!(
        issued.matfile_authority().expect("owner").lifetime,
        ForeignLifetime::Persistent
    );
}
