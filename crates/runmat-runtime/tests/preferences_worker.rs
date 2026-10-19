#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use futures::channel::oneshot;
use js_sys::Math;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_runtime::context::{
    BrowserPreferenceService, PreferenceServiceError, RuntimeContext, RuntimeServicePorts,
    UserPathMutation, UserPathServiceError, UserPathSetting,
};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_value::{IntValue, Value};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{Event, WorkerGlobalScope};

wasm_bindgen_test_configure!(run_in_worker);

fn worker_scope() -> WorkerGlobalScope {
    assert!(
        web_sys::window().is_none(),
        "the settings test must run in a Worker"
    );
    js_sys::global()
        .dyn_into::<WorkerGlobalScope>()
        .expect("WorkerGlobalScope")
}

fn context_with(service: Rc<BrowserPreferenceService>) -> RuntimeContext {
    let ports = RuntimeServicePorts::default()
        .with_preferences(service.clone())
        .with_user_path(service);
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(ports)
}

async fn delete_database_in_worker(database_name: &str) {
    let factory = worker_scope()
        .indexed_db()
        .expect("Worker IndexedDB access")
        .expect("Worker IndexedDB factory");
    let request = factory
        .delete_database(database_name)
        .expect("delete isolated settings database");
    let (sender, receiver) = oneshot::channel();
    let sender = Rc::new(RefCell::new(Some(sender)));

    let success_sender = sender.clone();
    let on_success = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = success_sender.borrow_mut().take() {
            let _ = sender.send(Ok(()));
        }
    });
    request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));

    let error_sender = sender.clone();
    let on_error = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = error_sender.borrow_mut().take() {
            let _ = sender.send(Err("Worker settings database deletion failed"));
        }
    });
    request.set_onerror(Some(on_error.as_ref().unchecked_ref()));

    let on_blocked = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Some(sender) = sender.borrow_mut().take() {
            let _ = sender.send(Err("a Worker settings connection remained open"));
        }
    });
    request.set_onblocked(Some(on_blocked.as_ref().unchecked_ref()));

    let outcome = receiver.await.expect("database deletion terminal event");
    request.set_onsuccess(None);
    request.set_onerror(None);
    request.set_onblocked(None);
    outcome.expect("all settings connections were closed");
}

#[wasm_bindgen_test(async)]
async fn worker_settings_reopen_exact_values_and_revisions_without_leaking_connections() {
    let _scope = worker_scope();
    let namespace = format!(
        "worker-preferences-{}-{}",
        js_sys::Date::now(),
        Math::random()
    );
    let database_name = format!("runmat.application.settings.v1/{namespace}");
    let adapter = Rc::new(BrowserPreferenceService::new(namespace.clone()).unwrap());
    let context = context_with(adapter.clone());
    let preferences = Rc::clone(
        context
            .service_ports()
            .require_preferences("Worker preferences")
            .unwrap(),
    );
    let user_path = Rc::clone(
        context
            .service_ports()
            .require_user_path("Worker user path")
            .unwrap(),
    );

    let initial = user_path.read(context.clone()).await.unwrap();
    assert_eq!(initial.setting, UserPathSetting::Default);
    assert_eq!(initial.revision, 0);
    assert!(preferences
        .all_groups(context.clone())
        .await
        .unwrap()
        .is_empty());

    preferences
        .set_many(
            context.clone(),
            "worker".into(),
            vec![("wide".into(), Value::Int(IntValue::U64(u64::MAX)))],
        )
        .await
        .unwrap();
    assert_eq!(
        user_path.read(context.clone()).await.unwrap().revision,
        0,
        "preference writes must not advance the distinct user-path revision"
    );
    let custom = user_path
        .commit(
            context.clone(),
            UserPathMutation::Set("relative/MiXeD".into()),
        )
        .await
        .unwrap();
    assert_eq!(
        custom.setting,
        UserPathSetting::Custom("relative/MiXeD".into())
    );
    assert_eq!(custom.revision, 1);

    let cloned_context = context.clone();
    let cloned_preferences = Rc::clone(
        cloned_context
            .service_ports()
            .require_preferences("cloned Worker session")
            .unwrap(),
    );
    assert_eq!(
        cloned_preferences
            .group_values(cloned_context, "worker".into())
            .await
            .unwrap()
            .unwrap()["wide"],
        Value::Int(IntValue::U64(u64::MAX))
    );

    let reopened = Rc::new(BrowserPreferenceService::new(namespace).unwrap());
    let fresh = context_with(reopened.clone());
    let fresh_preferences = Rc::clone(
        fresh
            .service_ports()
            .require_preferences("new Worker session")
            .unwrap(),
    );
    let fresh_path = Rc::clone(
        fresh
            .service_ports()
            .require_user_path("new Worker session")
            .unwrap(),
    );
    assert_eq!(fresh_path.read(fresh.clone()).await.unwrap(), custom);
    assert_eq!(
        fresh_preferences
            .group_values(fresh.clone(), "worker".into())
            .await
            .unwrap()
            .unwrap()["wide"],
        Value::Int(IntValue::U64(u64::MAX))
    );

    let cleared = fresh_path
        .commit(fresh.clone(), UserPathMutation::Clear)
        .await
        .unwrap();
    assert_eq!(cleared.setting, UserPathSetting::Cleared);
    assert_eq!(cleared.revision, 2);
    let reset = fresh_path
        .commit(fresh.clone(), UserPathMutation::Reset)
        .await
        .unwrap();
    assert_eq!(reset.setting, UserPathSetting::Default);
    assert_eq!(reset.revision, 3);

    assert!(matches!(
        fresh_path
            .commit(fresh.clone(), UserPathMutation::Set(String::new()))
            .await,
        Err(UserPathServiceError::EmptyCustomPath)
    ));
    assert!(matches!(
        fresh_preferences
            .read_or_create_many(
                fresh.clone(),
                "worker".into(),
                vec!["bad".into()],
                Some(vec![])
            )
            .await,
        Err(PreferenceServiceError::InvalidTransaction(_))
    ));
    drop(fresh_preferences.set_many(
        fresh.clone(),
        "worker".into(),
        vec![("unpolled".into(), Value::Num(1.0))],
    ));
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        fresh_preferences
            .set_many(
                fresh.clone().with_cancellation_token(cancellation.token()),
                "worker".into(),
                vec![("cancelled".into(), Value::Num(2.0))],
            )
            .await,
        Err(PreferenceServiceError::Execution(_))
    ));
    let values = fresh_preferences
        .group_values(fresh.clone(), "worker".into())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values["wide"], Value::Int(IntValue::U64(u64::MAX)));
    assert_eq!(fresh_path.read(fresh.clone()).await.unwrap(), reset);

    drop(fresh_path);
    drop(fresh_preferences);
    drop(fresh);
    drop(reopened);
    drop(user_path);
    drop(preferences);
    drop(context);
    drop(adapter);
    delete_database_in_worker(&database_name).await;
}
