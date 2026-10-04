use super::*;
use crate::execution::RuntimeExecutionService;
use futures::channel::oneshot;
use js_sys::Uint8Array;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::IntValue;
use std::cell::RefCell;
use std::future::Future;
use std::task::{Context, Poll};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::{Event, IdbTransactionMode};

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn unique_service() -> BrowserPreferenceService {
    let suffix = format!("test-{}-{}", js_sys::Date::now(), js_sys::Math::random());
    BrowserPreferenceService::new(suffix).unwrap()
}

async fn delete_namespace(service: &BrowserPreferenceService) {
    let request = connection::factory()
        .unwrap()
        .delete_database(&service.namespace)
        .unwrap();
    let (sender, receiver) = oneshot::channel();
    let sender = Rc::new(RefCell::new(Some(sender)));
    let success_sender = sender.clone();
    let on_success = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = success_sender.borrow_mut().take() {
            let _ = sender.send(Ok(()));
        }
    });
    request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
    let failed_sender = sender.clone();
    let on_error = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = failed_sender.borrow_mut().take() {
            let _ = sender.send(Err("delete failed"));
        }
    });
    request.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    let on_blocked = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = sender.borrow_mut().take() {
            let _ = sender.send(Err("a settings connection remained open"));
        }
    });
    request.set_onblocked(Some(on_blocked.as_ref().unchecked_ref()));
    receiver.await.unwrap().unwrap();
    request.set_onsuccess(None);
    request.set_onerror(None);
    request.set_onblocked(None);
}

async fn store_invalid_document(service: &BrowserPreferenceService) {
    let database = connection::open(&service.namespace).await.unwrap();
    let transaction = database
        .transaction_with_str_and_mode(connection::DOCUMENT_STORE, IdbTransactionMode::Readwrite)
        .unwrap();
    let store = transaction
        .object_store(connection::DOCUMENT_STORE)
        .unwrap();
    let bytes = Uint8Array::from(b"{invalid".as_slice());
    store
        .put_with_key(&bytes.into(), &JsValue::from_str(connection::DOCUMENT_KEY))
        .unwrap();
    let (sender, receiver) = oneshot::channel();
    let sender = Rc::new(RefCell::new(Some(sender)));
    let on_complete = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = sender.borrow_mut().take() {
            let _ = sender.send(());
        }
    });
    transaction.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
    receiver.await.unwrap();
    transaction.set_oncomplete(None);
}

#[wasm_bindgen_test(async)]
async fn indexeddb_reopens_exact_values_and_distinct_user_path_in_current_scope() {
    let service = unique_service();
    let initial = RuntimeUserPathService::read(&service, runtime())
        .await
        .unwrap();
    assert_eq!(initial.setting, crate::context::UserPathSetting::Default);
    assert_eq!(initial.revision, 0);
    service
        .set_many(
            runtime(),
            "app".into(),
            vec![("wide".into(), Value::Int(IntValue::U64(u64::MAX)))],
        )
        .await
        .unwrap();
    let custom = RuntimeUserPathService::commit(
        &service,
        runtime(),
        UserPathMutation::Set("relative/MiXeD".into()),
    )
    .await
    .unwrap();
    assert_eq!(custom.revision, 1);
    let reopened = BrowserPreferenceService::new(
        service
            .namespace
            .strip_prefix("runmat.application.settings.v1/")
            .unwrap()
            .to_string(),
    )
    .unwrap();
    assert_eq!(
        RuntimeUserPathService::read(&reopened, runtime())
            .await
            .unwrap(),
        custom
    );
    assert_eq!(
        reopened
            .group_values(runtime(), "app".into())
            .await
            .unwrap()
            .unwrap()["wide"],
        Value::Int(IntValue::U64(u64::MAX))
    );
    delete_namespace(&service).await;
}

#[wasm_bindgen_test(async)]
async fn unpolled_invalid_and_cancelled_calls_cannot_publish() {
    let service = unique_service();
    drop(service.set_many(
        runtime(),
        "app".into(),
        vec![("unpolled".into(), Value::Num(1.0))],
    ));
    assert!(matches!(
        RuntimeUserPathService::commit(&service, runtime(), UserPathMutation::Set(String::new()))
            .await,
        Err(UserPathServiceError::EmptyCustomPath)
    ));
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let cancelled = runtime().with_cancellation_token(cancellation.token());
    assert!(matches!(
        service
            .set_many(
                cancelled,
                "app".into(),
                vec![("cancelled".into(), Value::Num(2.0))]
            )
            .await,
        Err(PreferenceServiceError::Execution(
            crate::execution::ExecutionServiceError::Cancelled
        ))
    ));
    assert!(service.all_groups(runtime()).await.unwrap().is_empty());
    delete_namespace(&service).await;
}

#[wasm_bindgen_test(async)]
async fn blocked_upgrade_resumes_and_dropped_open_closes_late_success() {
    let resumed = unique_service();
    let held = connection::open(&resumed.namespace).await.unwrap();
    let (blocked_tx, blocked_rx) = oneshot::channel();
    let mut upgrade = Box::pin(connection::open_version(
        &resumed.namespace,
        2,
        Some(blocked_tx),
    ));
    assert!(matches!(
        Future::poll(
            upgrade.as_mut(),
            &mut Context::from_waker(futures::task::noop_waker_ref())
        ),
        Poll::Pending
    ));
    blocked_rx
        .await
        .expect("the held version-one connection blocks upgrade");
    drop(held);
    let upgraded = upgrade.await.unwrap();
    drop(upgraded);
    delete_namespace(&resumed).await;

    let abandoned = unique_service();
    let held = connection::open(&abandoned.namespace).await.unwrap();
    let (blocked_tx, blocked_rx) = oneshot::channel();
    let mut upgrade = Box::pin(connection::open_version(
        &abandoned.namespace,
        2,
        Some(blocked_tx),
    ));
    assert!(matches!(
        Future::poll(
            upgrade.as_mut(),
            &mut Context::from_waker(futures::task::noop_waker_ref())
        ),
        Poll::Pending
    ));
    blocked_rx
        .await
        .expect("the abandoned upgrade was actually blocked");
    drop(upgrade);
    drop(held);
    // The host-owned observer must close a successful request whose caller
    // vanished; otherwise deleting this unique namespace reports blocked.
    delete_namespace(&abandoned).await;
}

#[wasm_bindgen_test(async)]
async fn dropped_authorized_writer_keeps_one_terminal_owner_and_closes_connection() {
    let service = unique_service();
    let database = connection::open(&service.namespace).await.unwrap();
    let gate = service.gate.clone().lock_owned().await;
    let mutation = DocumentMutation::Set {
        group: "app".into(),
        entries: vec![(
            "after-drop".into(),
            encode_durable_value(&Value::Num(7.0)).unwrap(),
        )],
    };
    let mut write = Box::pin(transaction::mutate(database, mutation, gate));
    assert!(matches!(
        Future::poll(
            write.as_mut(),
            &mut Context::from_waker(futures::task::noop_waker_ref())
        ),
        Poll::Pending
    ));
    drop(write);
    let terminal_gate = service.gate.clone().lock_owned().await;
    drop(terminal_gate);
    assert_eq!(
        service
            .group_values(runtime(), "app".into())
            .await
            .unwrap()
            .unwrap()["after-drop"],
        Value::Num(7.0)
    );
    delete_namespace(&service).await;
}

#[wasm_bindgen_test(async)]
async fn late_missing_name_aborts_private_document_without_partial_publication() {
    let service = unique_service();
    let database = connection::open(&service.namespace).await.unwrap();
    let gate = service.gate.clone().lock_owned().await;
    let result = transaction::mutate(
        database,
        DocumentMutation::ReadOrCreate {
            group: "app".into(),
            names: vec!["would-insert".into(), "missing".into()],
            defaults: Some(vec![encode_durable_value(&Value::Num(1.0)).unwrap()]),
        },
        gate,
    )
    .await;
    assert!(matches!(
        result,
        Err(BrowserPreferenceError::Mutation(
            DocumentMutationError::Missing { name, .. }
        )) if name == "missing"
    ));
    assert!(service.all_groups(runtime()).await.unwrap().is_empty());
    delete_namespace(&service).await;
}

#[wasm_bindgen_test(async)]
async fn cancelled_blocked_open_and_corrupt_read_release_exact_connection() {
    let cancelled = unique_service();
    let held = connection::open(&cancelled.namespace).await.unwrap();
    let (blocked_tx, blocked_rx) = oneshot::channel();
    let cancellation = CancellationSource::new();
    let cancelled_runtime = runtime().with_cancellation_token(cancellation.token());
    let mut budget = cancelled_runtime.cpu_cooperation();
    let mut pending = Box::pin(budget.await_cancellable(connection::open_version(
        &cancelled.namespace,
        2,
        Some(blocked_tx),
    )));
    assert!(matches!(
        Future::poll(
            pending.as_mut(),
            &mut Context::from_waker(futures::task::noop_waker_ref())
        ),
        Poll::Pending
    ));
    blocked_rx.await.expect("upgrade reached blocked event");
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        pending.await,
        Err(crate::execution::ExecutionServiceError::Cancelled)
    ));
    drop(held);
    delete_namespace(&cancelled).await;

    let corrupt = unique_service();
    store_invalid_document(&corrupt).await;
    assert!(matches!(
        corrupt.all_groups(runtime()).await,
        Err(PreferenceServiceError::Document(_))
    ));
    delete_namespace(&corrupt).await;
}
