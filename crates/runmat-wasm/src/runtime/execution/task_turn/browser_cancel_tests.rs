//! Public browser-execution cancellation after an actual posted CPU task turn.

use super::TEST_POSTED_TURNS;
use js_sys::{Array, Reflect};
use serde::Serialize;
use serde_json::json;
use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;
use std::task::Poll;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::{MessageChannel, MessageEvent};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn public_workspace_clear_and_reset_keep_the_browser_session_usable() {
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    let runtime = crate::init_runmat(JsValue::NULL)
        .await
        .expect("initialize browser runtime");
    let request = json!({
        "source": { "kind": "text", "name": "workspace-clear.m", "text": "retained = uint16(7);" },
        "compatibility": "runmat"
    })
    .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    .expect("encode execution request");
    runtime
        .execute_request_js(request)
        .await
        .expect("execute workspace assignment");
    let snapshot = runtime
        .workspace_snapshot()
        .expect("read populated workspace");
    let values = Reflect::get(&snapshot, &JsValue::from_str("values")).expect("workspace values");
    assert_eq!(Array::from(&values).length(), 1);

    runtime.clear_workspace().expect("clear workspace");
    let snapshot = runtime
        .workspace_snapshot()
        .expect("read cleared workspace");
    let values =
        Reflect::get(&snapshot, &JsValue::from_str("values")).expect("cleared workspace values");
    assert_eq!(Array::from(&values).length(), 0);

    let previous_owner = runtime.session.borrow().browser_local_task_supervisor();
    runtime.reset_session().expect("reset browser session");
    assert!(!Rc::ptr_eq(
        &previous_owner,
        &runtime.session.borrow().browser_local_task_supervisor(),
    ));
    runtime.workspace_snapshot().expect("read reset workspace");
    runtime.dispose();
    runtime
        .clear_workspace()
        .expect("disposed clear remains idempotent");
    assert!(runtime.workspace_snapshot().is_err());
}

async fn cancel_during_public_execution(name: &str, text: &str, result_name: &str) {
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    let runtime = Rc::new(
        crate::init_runmat(JsValue::NULL)
            .await
            .expect("initialize browser runtime"),
    );
    let request = json!({
        "source": { "kind": "text", "name": name, "text": text },
        "compatibility": "runmat"
    })
    .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    .expect("encode execution request");
    let turns_before = TEST_POSTED_TURNS.with(std::cell::Cell::get);
    let mut execution = Box::pin(runtime.execute_request_js(request));
    futures::future::poll_fn(|context| match execution.as_mut().poll(context) {
        Poll::Pending => {
            if TEST_POSTED_TURNS.with(std::cell::Cell::get) > turns_before {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }
        Poll::Ready(Ok(payload)) => {
            let diagnostic = js_sys::JSON::stringify(&payload)
                .ok()
                .and_then(|value| value.as_string());
            panic!("{name} completed before its cooperative turn: {diagnostic:?}");
        }
        Poll::Ready(Err(error)) => {
            panic!("{name} failed before its cooperative turn: {error:?}");
        }
    })
    .await;

    // A separate task on this event loop delivers cancellation while the
    // public execution future is suspended at its own posted turn.
    let channel = MessageChannel::new().expect("browser cancellation task channel");
    let receive = channel.port1();
    let send = channel.port2();
    let (canceled_tx, canceled_rx) = futures::channel::oneshot::channel();
    let canceled_tx = Rc::new(RefCell::new(Some(canceled_tx)));
    let cancel_runtime = Rc::clone(&runtime);
    let callback = Closure::wrap(Box::new(move |_event: MessageEvent| {
        cancel_runtime.cancel_execution();
        if let Some(sender) = canceled_tx.borrow_mut().take() {
            let _ = sender.send(());
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    receive.set_onmessage(Some(callback.as_ref().unchecked_ref()));
    receive.start();
    send.post_message(&JsValue::UNDEFINED)
        .expect("queue same-event-loop cancellation task");
    canceled_rx.await.expect("cancellation task must run");
    receive.set_onmessage(None);
    receive.close();
    send.close();
    drop(callback);

    let payload = execution
        .await
        .expect("execution returns a structured result");
    let error = Reflect::get(&payload, &JsValue::from_str("error")).expect("execution error field");
    let identifier = Reflect::get(&error, &JsValue::from_str("identifier"))
        .expect("structured cancellation identifier");
    // The VM drops the suspended builtin before repolling it. Its public
    // cancellation takes precedence over the builtin-local typed cause.
    assert_eq!(
        identifier.as_string().as_deref(),
        Some("RunMat:ExecutionCancelled")
    );
    let message = Reflect::get(&error, &JsValue::from_str("message"))
        .expect("structured cancellation message");
    assert_eq!(
        message.as_string().as_deref(),
        Some("Execution cancelled by user")
    );

    // The result assignment and subsequent statement must not publish after
    // the suspended operation is abandoned.
    let snapshot = runtime.workspace_snapshot().expect("workspace snapshot");
    let values = Array::from(
        &Reflect::get(&snapshot, &JsValue::from_str("values")).expect("workspace values"),
    );
    for entry in values.iter() {
        let entry_name = Reflect::get(&entry, &JsValue::from_str("name"))
            .expect("workspace entry name")
            .as_string()
            .expect("workspace name text");
        assert_ne!(entry_name, result_name, "cancelled result must not publish");
        assert_ne!(
            entry_name, "completed",
            "post-call statement must not replay"
        );
    }
}

#[wasm_bindgen_test(async)]
async fn same_browser_event_loop_cancel_interrupts_a_running_multinomial_fit() {
    cancel_during_public_execution(
        "mnrfit-browser-cancel.m",
        "X = (1:65536)'; Y = mod(X,2)+1; B = mnrfit(X,Y,'IterationLimit',100000); completed = 1;",
        "B",
    )
    .await;
}

#[wasm_bindgen_test(async)]
async fn same_browser_event_loop_cancel_interrupts_running_kmeans() {
    cancel_during_public_execution(
        "kmeans-browser-cancel.m",
        "X = (1:65536)'; idx = kmeans(X,2,'Start',[1;65536],'MaxIter',100000); completed = 1;",
        "idx",
    )
    .await;
}

#[wasm_bindgen_test(async)]
async fn same_browser_event_loop_cancel_interrupts_running_tsne() {
    cancel_during_public_execution(
        "tsne-browser-cancel.m",
        "X = [(1:512)', mod((1:512)',7)]; embedding = tsne(X,'NumDimensions',2,'Perplexity',5); completed = 1;",
        "embedding",
    )
    .await;
}

#[wasm_bindgen_test(async)]
async fn dispose_while_public_execution_is_pending_never_restores_its_task_owner() {
    let runtime = Rc::new(
        crate::init_runmat(JsValue::NULL)
            .await
            .expect("initialize browser runtime"),
    );
    let original_owner = runtime.session.borrow().browser_local_task_supervisor();
    let request = json!({
        "source": { "kind": "text", "name": "dispose-pending.m", "text": "pause(1); completed = 1;" },
        "compatibility": "runmat"
    })
    .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    .expect("encode execution request");
    let mut execution = Box::pin(runtime.execute_request_js(request.clone()));
    futures::future::poll_fn(|context| match execution.as_mut().poll(context) {
        Poll::Pending => Poll::Ready(()),
        Poll::Ready(result) => panic!("execution must suspend before disposal: {result:?}"),
    })
    .await;

    assert!(runtime.active_interrupt.borrow().is_some());
    assert!(
        runtime.execute_request_js(request.clone()).await.is_err(),
        "second request must reject before replacing active cancellation"
    );
    assert!(
        runtime.active_interrupt.borrow().is_some(),
        "rejected request must not clear first request's cancel authority"
    );
    assert!(runtime.workspace_snapshot().is_err());
    assert!(runtime.stats().is_err());
    runtime.dispose();
    assert!(
        runtime
            .active_interrupt
            .borrow()
            .as_ref()
            .is_some_and(|active| active.token().is_cancelled()),
        "dispose cancels the original in-flight source"
    );
    let _ = execution
        .await
        .expect("in-flight request returns a structured result");
    assert!(
        !Rc::ptr_eq(
            &original_owner,
            &runtime.session.borrow().browser_local_task_supervisor()
        ),
        "disposed in-flight session must not replace its placeholder"
    );
    assert!(
        runtime.execute_request_js(request).await.is_err(),
        "disposed host cannot admit another execution"
    );
}
