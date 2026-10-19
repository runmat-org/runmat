use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderPrecision,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingProvider(AtomicUsize);
static COUNTER: CountingProvider = CountingProvider(AtomicUsize::new(0));

impl AccelProvider for CountingProvider {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.0.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("preflight must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("preflight must not download") })
    }
    fn device_info(&self) -> String {
        "preflight counter".into()
    }
}

#[test]
fn rejected_complete_epilogues_do_not_upload() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&COUNTER));
        COUNTER.0.store(0, Ordering::SeqCst);
        for expression in [
            "sin(A*B)",
            "(A*B)+C",
            "((A*B)+2)*3",
            "2./(A*B)",
            "((A*B).*c).*r",
            "(A*B).^0.5",
            "max(min(A*B,7),3)",
        ] {
            assert!(
                matches!(direct(expression), FusionExecutionOutcome::SharedRuntime(_)),
                "{expression}"
            );
            assert_eq!(COUNTER.0.load(Ordering::SeqCst), 0, "{expression}");
        }
    });
}

#[test]
fn selected_named_override_prevents_upload_before_matrix_preparation() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&COUNTER));
        COUNTER.0.store(0, Ordering::SeqCst);
        let (mut plan, inputs) = fixture::requested("A*B", None);
        let runmat_accelerate::fusion::FusionOp::Builtin { origin, .. } = &mut plan.operations[0]
        else {
            panic!("compiled matrix root");
        };
        *origin = runmat_accelerate::graph::FusionCallOrigin::named("mtimes");
        let result = block_on(execute_matmul_epilogue_placed(FusionExecutionRequest {
            plan: &plan,
            inputs,
            placement: None,
            runtime: Some(selected_binding::runtime()),
        }))
        .unwrap();
        assert!(matches!(result, FusionExecutionOutcome::SharedRuntime(_)));
        assert_eq!(COUNTER.0.load(Ordering::SeqCst), 0);
    });
}
