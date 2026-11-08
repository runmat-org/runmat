//! Two actual child processes retain source-local success/failure console evidence.
use super::*;
use runmat_execution::console::{ConsoleBudget, ConsoleCompletion, ConsoleLimits, ConsoleStream};
use runmat_execution::identity::AttemptId;
use runmat_execution::PoolId;
use runmat_runtime::context::{
    PortableConsoleHost, RuntimeConsoleScopeIdentity, RuntimeHostService,
};

pub(super) async fn verify(config: &NativeExecutionConfig) {
    let mut config = config.clone();
    config.max_workers = 1;
    let budget = ConsoleBudget {
        max_scopes: 16,
        max_records: 32,
        max_text_bytes: 4096,
        max_encoded_bytes: 65536,
        max_retained_bytes: 131072,
    };
    let owner = runmat_execution_runner_native::console_admission_from_subgrant(budget).unwrap();
    let host = Rc::new(
        PortableConsoleHost::new(ConsoleLimits {
            max_records: 32,
            max_text_bytes: 4096,
        })
        .unwrap(),
    );
    host.install_console_admission(
        owner,
        RuntimeConsoleScopeIdentity {
            scope_id: ExecutionScopeId::derive(&[b"outer-console"]),
            pool_id: PoolId::derive(&[b"outer-console"]),
            task_id: runmat_execution::TaskId::derive(&[b"outer-console"]),
            attempt_id: AttemptId::derive(&[b"outer-console"]),
        },
    )
    .unwrap();
    host.console(ConsoleStream::Stdout, "parent-only".into());
    let service = NativeExecutionService::new(config).unwrap();
    let mut tasks = Vec::new();
    for (fails, source) in [
        (false, "function [first, second] = original_output_values(); fprintf(1,'λ🙂 child success'); first=42; second=0; end"),
        (true, "function [first, second] = original_output_values(); fprintf(2,'λ🙂 child failure prefix'); error('RunMat:OriginalChildFailure','expected child failure'); first=42; second=0; end"),
    ] {
        let mut original = call_source(1, source).await;
        original.console = Some(ConsoleLimits { max_records: 8, max_text_bytes: 1000 });
        let ports = original.runtime.service_ports().clone().with_host(host.clone());
        let runtime = original.runtime.clone().with_service_ports(ports);
        let future = service.create_future(original.call.clone()).unwrap();
        let task = service.spawn_in_context(&future, None, &runtime).unwrap();
        tasks.push(task.clone());
        let result = await_in_context(&service, Value::Task(task.into()), &runtime).await;
        if fails {
            assert!(result.unwrap_err().to_string().contains("expected child failure"));
        } else {
            assert_eq!(result.unwrap().value().iter().cloned().collect::<Vec<_>>(), vec![Value::Num(42.0)]);
        }
    }
    service.drain_scope(CancellationReason::Shutdown);
    let retained = host.retained_console_source();
    let root = host.finish(false).unwrap();
    drop(service);
    drop(host);
    assert_eq!(
        root.records().len(),
        1,
        "root console: {:?}",
        root.records()
    );
    assert_eq!(root.records()[0].text.as_ref(), "parent-only");
    let mut identities = BTreeSet::new();
    let mut texts = BTreeSet::new();
    retained
        .visit(&mut |scope| {
            assert!(tasks.iter().any(|task| task.id == scope.identity.task_id
                && task.scope_id == scope.identity.scope_id));
            assert!(identities.insert((scope.identity.task_id, scope.identity.attempt_id)));
            scope.transcript.validate().unwrap();
            assert_eq!(scope.transcript.records().len(), 1);
            assert_eq!(scope.transcript.records()[0].sequence, 1);
            assert_eq!(scope.transcript.completion(), ConsoleCompletion::Complete);
            assert!(scope.transcript.source_receipt().is_some());
            texts.insert(scope.transcript.records()[0].text.to_string());
            Ok(())
        })
        .unwrap();
    assert_eq!(identities.len(), 2);
    assert_eq!(
        texts,
        BTreeSet::from([
            "λ🙂 child success".to_string(),
            "λ🙂 child failure prefix".to_string()
        ])
    );
}
