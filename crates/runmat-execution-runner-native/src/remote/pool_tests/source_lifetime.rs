use super::*;
use runmat_runtime::context::{
    RuntimeCollectiveService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use runmat_types::{LabCount, LabRank, ProgramSourceId};
use std::future::Future as _;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

struct PendingCollective {
    task: runmat_execution::SpmdTaskContext,
    calls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    selected_sources: Arc<std::sync::Mutex<Vec<runmat_runtime::source_context::RuntimeSourceRef>>>,
}

struct PendingCollectiveOwner(Arc<AtomicUsize>);

impl Drop for PendingCollectiveOwner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl RuntimeCollectiveService for PendingCollective {
    fn context(&self) -> &runmat_execution::SpmdTaskContext {
        &self.task
    }

    fn next_sequence(
        &self,
        _: runmat_types::CollectiveId,
    ) -> Result<runmat_execution::CollectiveSequence, runmat_runtime::RuntimeError> {
        Ok(runmat_execution::CollectiveSequence(1))
    }

    fn execute(
        &self,
        _: runmat_execution::CollectiveRequest,
    ) -> RuntimeServiceFuture<
        Result<runmat_execution::CollectiveResponse, runmat_runtime::RuntimeError>,
    > {
        let selected = runmat_runtime::context::legacy::active()
            .and_then(|runtime| runtime.program_source_ref(ProgramSourceId(0)))
            .expect("pending collective runs in the selected worker source binding");
        self.selected_sources
            .lock()
            .expect("source recorder lock")
            .push(selected);
        self.calls.fetch_add(1, Ordering::SeqCst);
        let owner = PendingCollectiveOwner(Arc::clone(&self.drops));
        Box::pin(async move {
            let _owner = owner;
            std::future::pending().await
        })
    }
}

#[tokio::test]
async fn compiled_region_worker_drop_retires_source_and_pending_effect_without_replay() {
    let program = compile_spmd_region_bundle("spmd\n  spmdBarrier();\nend\n", 0).await;
    let scope_id = ExecutionScopeId::derive(&[b"source-lifetime-scope"]);
    let pool = PoolHandle {
        id: PoolId::derive(&[b"source-lifetime-pool"]),
        scope_id,
        generation: 1,
    };
    let task = runmat_execution::SpmdTaskContext {
        gang: GangHandle {
            id: GangId::derive(&[b"source-lifetime-gang"]),
            scope_id,
            generation: 1,
            pool,
            labs: LabCount(1),
        },
        region: program.region,
        rank: LabRank(1),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let selected_sources = Arc::new(std::sync::Mutex::new(Vec::new()));
    let collective = Rc::new(PendingCollective {
        task: task.clone(),
        calls: Arc::clone(&calls),
        drops: Arc::clone(&drops),
        selected_sources: Arc::clone(&selected_sources),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_collective(collective));
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: program.recipe,
        artifact: program.artifact,
        callable: ProgramCallable::spmd_region(program.region),
        context: ProgramInvocationContext::SpmdTask { task },
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 0,
    };
    request.validate_for_portable_host().unwrap();
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    for invocation in 1..=2 {
        let mut worker = Box::pin(runmat_vm::execute_program_request_with_context(
            request.clone(),
            runtime.clone(),
        ));
        for _ in 0..64 {
            assert!(matches!(worker.as_mut().poll(&mut cx), Poll::Pending));
            if calls.load(Ordering::SeqCst) == invocation {
                break;
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), invocation);
        assert_eq!(drops.load(Ordering::SeqCst), invocation - 1);
        let recorded = selected_sources.lock().expect("source recorder lock");
        assert_eq!(recorded.len(), invocation);
        let issued = recorded[invocation - 1];
        if invocation == 2 {
            assert_ne!(issued, recorded[0]);
            assert_eq!(recorded[0].resolve_in(&runtime), None);
        }
        drop(recorded);
        assert_eq!(issued.resolve_in(&runtime), Some(issued.source_id()));
        assert_eq!(
            runmat_runtime::source_context::source_info_in_context(&runtime, issued.source_id())
                .unwrap()
                .text
                .as_ref(),
            "spmd\n  spmdBarrier();\nend\n"
        );
        drop(worker);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            invocation,
            "abandonment must not replay a collective"
        );
        assert_eq!(drops.load(Ordering::SeqCst), invocation);
        assert_eq!(issued.resolve_in(&runtime), None);
        assert!(runmat_runtime::source_context::source_info_in_context(
            &runtime,
            issued.source_id()
        )
        .is_none());
    }
}
