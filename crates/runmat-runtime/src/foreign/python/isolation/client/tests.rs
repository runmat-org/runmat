use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use super::*;
use crate::execution::RuntimeExecutionService;

#[tokio::test]
async fn closed_nested_request_never_writes_to_the_child() {
    let mut client = IsolatedPythonClient::pending_test_client().await;
    let handles = ForeignHandleRegistry::default();
    let nested = Rc::new(RefCell::new(VecDeque::new()));
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let (reply, response) = oneshot::channel();
    drop(response);
    let queued = NestedPythonCall {
        call: ForeignCall {
            adapter: "python".into(),
            symbol: "must-not-run".into(),
            arguments: crate::call::RootedFunctionArguments::ordinary(
                crate::execution::RootedValues::new(Vec::new()).unwrap(),
            ),
            requested_outputs: 1,
        },
        reply,
        ownership: Rc::new(()),
    };
    client
        .dispatch_nested_call(context, queued, &handles, "python-session", &nested)
        .await
        .unwrap();
    assert_eq!(client.next_request_id, 1);
    assert!(!client.nested_aborted);
}

#[tokio::test]
async fn dropping_dispatched_nested_request_terminalizes_owned_client() {
    let mut client = IsolatedPythonClient::pending_test_client().await;
    let pid = client.process_id().expect("owned pending child pid");
    let handles = ForeignHandleRegistry::default();
    let nested = Rc::new(RefCell::new(VecDeque::new()));
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let (reply, response) = oneshot::channel();
    let call = NestedPythonCall {
        call: ForeignCall {
            adapter: "python".into(),
            symbol: "status".into(),
            arguments: crate::call::RootedFunctionArguments::ordinary(
                crate::execution::RootedValues::new(Vec::new()).unwrap(),
            ),
            requested_outputs: 1,
        },
        reply,
        ownership: Rc::new(()),
    };
    let mut pending =
        Box::pin(client.dispatch_nested_call(context, call, &handles, "python-session", &nested));
    tokio::select! {
        result = &mut pending => panic!("sleeping child unexpectedly replied: {result:?}"),
        () = tokio::time::sleep(Duration::from_millis(20)) => {}
    }
    drop(response);
    assert_eq!(
        pending.await.unwrap_err().identifier(),
        Some("RunMat:Python:Cancelled")
    );
    assert!(client.nested_aborted);
    drop(client);
    let mut stopped = false;
    for _ in 0..100 {
        if unsafe { libc::kill(pid as libc::pid_t, 0) } == -1 {
            stopped = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(stopped, "abandoned nested host {pid} is still present");
}
