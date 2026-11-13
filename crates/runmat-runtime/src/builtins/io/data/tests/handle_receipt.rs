use super::*;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use runmat_filesystem::{self as fs, FsProvider};
use runmat_value::StructValue;
use std::{rc::Rc, sync::Arc};

fn fixture_root() -> std::path::PathBuf {
    std::path::PathBuf::from("/runmat-selected")
}

struct Finite(crate::context::host_effect::ManagedHostEffectLedger);
impl RuntimeParallelService for Finite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }

    fn managed_host_effect_ledger(
        &self,
    ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
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
    let rank_zero = Value::Tensor(Tensor::from_f32(vec![1.0], Vec::new()).unwrap());
    let matrix_scalar = Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
    assert_ne!(
        handle_admission::input(&rank_zero).identity_digest,
        handle_admission::input(&matrix_scalar).identity_digest,
        "exact identity includes the full low-rank shape"
    );
    let large = Value::Tensor(Tensor::from_f32(vec![0.0; 4_097], vec![1, 4_097]).unwrap());
    assert!(handle_admission::input(&large).identity_digest.is_none());
    let first_slice = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap());
    let second_slice = Value::Cell(CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
    assert_ne!(
        handle_admission::input(&first_slice).identity_digest,
        handle_admission::input(&second_slice).identity_digest
    );
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options.insert("if_manifest", Value::String("revision-1".into()));
    assert!(handle_admission::input(&Value::Struct(options.finish()))
        .identity_digest
        .is_some());
    assert!(
        handle_admission::input(&Value::String("x".repeat(1_048_577).into()))
            .identity_digest
            .is_none(),
        "oversized scalar text cannot monopolize admission"
    );
    let nested = Value::Cell(
        CellArray::new(vec![Value::String("x".repeat(1_048_577).into())], 1, 1).unwrap(),
    );
    assert!(handle_admission::input(&nested).identity_digest.is_none());
    let huge_chars = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars(vec!['x'; 4_097], 1, 4_097).unwrap(),
    );
    assert!(handle_admission::input(&huge_chars)
        .identity_digest
        .is_none());
    let mut whole_call = super::super::handle_identity::HashBudget::new();
    let first = handle_admission::input_with_budget(
        &Value::String("a".repeat(600_000).into()),
        &mut whole_call,
    );
    let second = handle_admission::input_with_budget(
        &Value::String("b".repeat(600_000).into()),
        &mut whole_call,
    );
    assert!(first.identity_digest.is_some());
    assert!(
        second.identity_digest.is_none(),
        "aggregate text work exceeds one call budget"
    );
    assert!(whole_call.is_exhausted());
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
fn ignored_trailing_receiver_controls_reject_before_storage_or_queue_effects() {
    use crate::dispatcher::call_builtin;

    let _serial = fs::provider_override_lock();
    let root = fixture_root();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir(&root));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);
    let mut meta = runmat_value::UnboundedStructValueBuilder::new();
    meta.insert("dtype", Value::String("f64".into()));
    meta.insert(
        "shape",
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays.insert("sample", Value::Struct(meta.finish()));
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema.insert("arrays", Value::Struct(arrays.finish()));
    let dataset = call_builtin(
        "data.create",
        &[
            Value::String("arity.data".into()),
            Value::Struct(schema.finish()),
        ],
    )
    .unwrap();
    let array = call_builtin(
        "Dataset.array",
        &[dataset.value().clone(), Value::String("sample".into())],
    )
    .unwrap();
    let transaction = call_builtin("Dataset.begin", std::slice::from_ref(dataset.value())).unwrap();
    let manifest_path = root.join("arity.data/manifest.json");
    let before = provider.read_project_path(&manifest_path).unwrap();
    for (name, arguments) in [
        (
            "Dataset.get_attr",
            vec![
                dataset.value().clone(),
                Value::String("k".into()),
                Value::Num(1.0),
                Value::Num(2.0),
            ],
        ),
        (
            "Dataset.begin",
            vec![dataset.value().clone(), Value::String("ignored".into())],
        ),
        (
            "Dataset.snapshot",
            vec![
                dataset.value().clone(),
                Value::String("label".into()),
                Value::String("ignored".into()),
            ],
        ),
        (
            "DataArray.read",
            vec![
                array.value().clone(),
                Value::String(":".into()),
                Value::String(":".into()),
            ],
        ),
        (
            "DataArray.resize",
            vec![
                array.value().clone(),
                Value::Num(2.0),
                Value::String("ignored".into()),
            ],
        ),
        (
            "DataArray.fill",
            vec![
                array.value().clone(),
                Value::Num(2.0),
                Value::String("ignored".into()),
            ],
        ),
        (
            "DataTransaction.write",
            vec![
                transaction.value().clone(),
                Value::String("sample".into()),
                Value::String(":".into()),
                Value::Num(2.0),
                Value::String("ignored".into()),
            ],
        ),
        (
            "DataTransaction.resize",
            vec![
                transaction.value().clone(),
                Value::String("sample".into()),
                Value::Num(2.0),
                Value::String("ignored".into()),
            ],
        ),
        (
            "DataTransaction.fill",
            vec![
                transaction.value().clone(),
                Value::String("sample".into()),
                Value::Num(2.0),
                Value::String(":".into()),
                Value::String("ignored".into()),
            ],
        ),
    ] {
        let error = call_builtin(name, &arguments).expect_err("unknown control must reject");
        assert!(!error.message().is_empty(), "{name}");
        assert_eq!(
            provider.read_project_path(&manifest_path).unwrap(),
            before,
            "{name}"
        );
    }
    assert_eq!(
        call_builtin(
            "DataTransaction.status",
            std::slice::from_ref(transaction.value())
        )
        .unwrap(),
        Value::String("open".into())
    );
    assert!(call_builtin(
        "DataTransaction.abort",
        std::slice::from_ref(transaction.value())
    )
    .is_ok());
}

#[test]
fn finite_uncertified_receiver_write_rejects_before_mutation() {
    let _serial = fs::provider_override_lock();
    let root = fixture_root();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir(&root));
    let _provider = fs::replace_provider(provider.clone() as Arc<dyn FsProvider>);
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(StructValue::new()));
    let dataset = call_builtin(
        "data.create",
        &[
            Value::String("owned.data".into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("context-free embedding creates a process-local managed handle");
    let before = provider
        .read_project_path(root.join("owned.data/manifest.json"))
        .expect("manifest before selection");
    let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(64 * 1_024);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger))));
    let error = futures::executor::block_on(crate::call_builtin_async_in_context(
        &runtime,
        "Dataset.set_attr",
        &[
            dataset.value().clone(),
            Value::String("owner".into()),
            Value::String("changed".into()),
        ],
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
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(StructValue::new()));
    let dataset = futures::executor::block_on(crate::call_builtin_async_in_context(
        &runtime,
        "data.create",
        &[
            Value::String("session.data".into()),
            Value::Struct(schema.finish()),
        ],
    ))
    .expect("session-issued Dataset");
    let next_invocation = runtime
        .clone()
        .with_cancellation_token(runmat_execution::cancellation::CancellationSource::new().token());
    assert_eq!(
        futures::executor::block_on(crate::call_builtin_async_in_context(
            &next_invocation,
            "Dataset.path",
            std::slice::from_ref(dataset.value())
        ))
        .unwrap(),
        Value::String("session.data".into())
    );
    let foreign = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let error = futures::executor::block_on(crate::call_builtin_async_in_context(
        &foreign,
        "Dataset.path",
        std::slice::from_ref(dataset.value()),
    ))
    .expect_err("separate session must not receive a sealed owner");
    assert_eq!(error.identifier.as_deref(), Some("RUNMAT:Data:Error"));
    drop(runtime);
    assert!(call_builtin("Dataset.path", std::slice::from_ref(dataset.value())).is_err());
}
