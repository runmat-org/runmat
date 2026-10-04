use super::*;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use runmat_filesystem::{self as fs, FsProvider};
use std::{rc::Rc, sync::Arc};

fn fixture_root() -> std::path::PathBuf {
    #[cfg(windows)]
    let root = std::path::PathBuf::from(r"C:\runmat-selected");
    #[cfg(not(windows))]
    let root = std::path::PathBuf::from("/runmat-selected");
    root
}

struct Finite;
impl RuntimeParallelService for Finite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(1_024),
            ..Default::default()
        }
    }
}

#[test]
fn handle_controls_do_not_alias_same_shape_literals_or_certify_unknown_payloads() {
    let first = handle_admission::input(&Value::String("first".into()));
    let second = handle_admission::input(&Value::String("second".into()));
    assert_eq!(first.fact, second.fact);
    assert_ne!(first.identity_digest, second.identity_digest);
    assert_ne!(
        handle_admission::input(&Value::Num(0.0)).identity_digest,
        handle_admission::input(&Value::Num(-0.0)).identity_digest
    );
    let payload = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let changed = Value::Tensor(Tensor::from_f32(vec![1.0, 3.0], vec![1, 2]).unwrap());
    assert_ne!(
        handle_admission::input(&payload).identity_digest,
        handle_admission::input(&changed).identity_digest,
        "same-class and same-shape native payloads need exact identity"
    );
    let large = Value::Tensor(Tensor::from_f32(vec![0.0; 4_097], vec![1, 4_097]).unwrap());
    assert!(handle_admission::input(&large).identity_digest.is_none());
    let first_slice = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap());
    let second_slice = Value::Cell(CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
    assert_ne!(
        handle_admission::input(&first_slice).identity_digest,
        handle_admission::input(&second_slice).identity_digest
    );
    let mut options = StructValue::new();
    options.insert("if_manifest", Value::String("revision-1".into()));
    assert!(handle_admission::input(&Value::Struct(options))
        .identity_digest
        .is_some());
}

#[test]
fn sealed_transaction_queue_revision_invalidates_prior_admission_identity() {
    let tx_id = crate::data::start_tx("queue.data".into(), 9).unwrap();
    let first = crate::data::with_tx(&tx_id, |tx| Ok(handle_admission::queue_input(tx)))
        .expect("new queue identity");
    crate::data::with_tx_mut(&tx_id, |tx| {
        tx.attrs.insert("version".into(), Value::Num(1.0));
        Ok(())
    })
    .expect("first queued mutation");
    let second = crate::data::with_tx(&tx_id, |tx| Ok(handle_admission::queue_input(tx)))
        .expect("changed queue identity");
    assert_ne!(first.identity_digest, second.identity_digest);
    crate::data::remove_tx(&tx_id).unwrap();
}

#[test]
fn finite_uncertified_receiver_write_rejects_before_mutation() {
    let _serial = fs::provider_override_lock();
    let root = fixture_root();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir(&root));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);
    let mut schema = StructValue::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(StructValue::new()));
    let dataset = futures::executor::block_on(data_create_builtin(
        None,
        Value::String("owned.data".into()),
        Value::Struct(schema),
    ))
    .expect("context-free embedding creates a process-local managed handle");
    let before = provider
        .read_project_path(root.join("owned.data/manifest.json"))
        .expect("manifest before selection");
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(Finite)));
    let error = futures::executor::block_on(dataset_set_attr_builtin(
        Some(runtime),
        dataset,
        Value::String("owner".into()),
        Value::String("changed".into()),
    ))
    .expect_err("finite receiver work needs certified whole-call resources");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(
        provider
            .read_project_path(root.join("owned.data/manifest.json"))
            .unwrap(),
        before,
        "rejected selection must not mutate the captured owner"
    );
}

#[test]
fn session_issued_dataset_survives_later_invocation_but_rejects_foreign_session() {
    let _serial = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir(fixture_root()));
    let _provider = fs::replace_provider(provider as Arc<dyn FsProvider>);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let mut schema = StructValue::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(StructValue::new()));
    let dataset = futures::executor::block_on(data_create_builtin(
        Some(runtime.clone()),
        Value::String("session.data".into()),
        Value::Struct(schema),
    ))
    .expect("session-issued Dataset");
    let next_invocation = runtime
        .clone()
        .with_cancellation_token(runmat_execution::cancellation::CancellationSource::new().token());
    assert_eq!(
        futures::executor::block_on(dataset_path_builtin(Some(next_invocation), dataset.clone(),))
            .unwrap(),
        Value::String("session.data".into())
    );
    let foreign = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let error = futures::executor::block_on(dataset_path_builtin(Some(foreign), dataset.clone()))
        .expect_err("separate session must not receive a sealed owner");
    assert_eq!(error.identifier.as_deref(), Some("RUNMAT:Data:Error"));
    drop(runtime);
    assert!(futures::executor::block_on(dataset_path_builtin(None, dataset)).is_err());
}
