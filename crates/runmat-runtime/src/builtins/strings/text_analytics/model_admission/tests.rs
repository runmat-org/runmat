use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::{ExecutionServiceError, RuntimeExecutionService};
use runmat_execution::cancellation::CancellationSource;
use std::cell::Cell;
use std::error::Error;
use std::rc::Rc;

struct Policy {
    finite: Cell<bool>,
    ledger: ManagedHostEffectLedger,
}
impl RuntimeParallelService for Policy {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.ledger.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: self.finite.get().then_some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
}

#[test]
fn finite_entry_and_continuation_refuse_before_publication_without_replay() {
    let ledger = ManagedHostEffectLedger::new(65536);
    let policy = Rc::new(Policy {
        finite: Cell::new(false),
        ledger: ledger.clone(),
    });
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_parallel(policy.clone()));
    let work = UnboundedModelWork::admit(Some(&runtime), "tokenizedDocument").unwrap();
    assert_eq!(work.finish(Ok(7)).unwrap(), 7);
    let polls = Cell::new(0);
    let before = ledger.snapshot().free_bytes();
    let error = futures::executor::block_on(work.wait(async {
        polls.set(polls.get() + 1);
        policy.finite.set(true);
        Ok(7)
    }))
    .unwrap_err();
    assert_eq!(polls.get(), 1);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(matches!(
        error
            .source()
            .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
        Some(ExecutionServiceError::Infrastructure(_))
    ));
    let error = futures::executor::block_on(super::super::documents::tokenized_document_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![runmat_value::Value::from(
            "alpha beta",
        )]),
    ))
    .unwrap_err();
    assert!(error
        .message()
        .contains("complete finite text-model workspace ownership is not implemented"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(ledger.snapshot().free_bytes(), before);
}

#[test]
fn stage_error_keeps_original_cause_and_disables_requested_replay() {
    let work = UnboundedModelWork::admit(None, "writeWordEmbedding").unwrap();
    let original = crate::build_runtime_error("physical write failed")
        .with_identifier("RunMat:test:PhysicalWrite")
        .with_source(std::io::Error::other("original physical cause"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
        .build();
    let error = work.finish::<()>(Err(original)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:test:PhysicalWrite"));
    assert_eq!(error.message(), "physical write failed");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error
        .source()
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .is_some());

    // Exercise the real canonical bindings with deliberately invalid model
    // operands. Output arity must fail before model parsing, gather or I/O.
    for (name, excess) in [
        ("tokenizedDocument", 2),
        ("bagOfWords", 2),
        ("removeShortWords", 2),
        ("removeLongWords", 2),
        ("removeWords", 2),
        ("removeStopWords", 2),
        ("word2ind", 2),
        ("ind2word", 2),
        ("isVocabularyWord", 2),
        ("doc2sequence", 2),
        ("word2vec", 2),
        ("vec2word", 3),
        ("cosineSimilarity", 2),
        ("writeWordEmbedding", 1),
    ] {
        let error = futures::executor::block_on(crate::call_builtin_async_with_outputs(
            name,
            &[runmat_value::Value::Num(0.0), runmat_value::Value::Num(0.0)],
            excess,
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str()),
            "{name}: {error}"
        );
    }
}
