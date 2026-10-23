use super::*;
use runmat_accelerate::{fusion::FusionOp, graph::FusionCallOrigin};
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture},
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use std::rc::Rc;

struct Override;
impl RuntimeBuiltinService for Override {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        let mut bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        for binding in &mut bindings {
            binding.implementation = replacement;
        }
        bindings
    }
}
fn replacement(_: runmat_runtime::builtin::RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        runmat_runtime::execution::RootedValueSequence::new(
            runmat_value::ValueSequence::single(Value::Num(99.))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        )
        .map_err(|error| error.into_runtime_error())
    })
}

pub(super) fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Override)))
}

#[test]
fn numeric_primitive_product_keeps_provider_route_under_named_override() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        let runtime = runtime();
        let (mut plan, inputs) = fixture::requested("A*B", None);
        assert!(matches!(
            &plan.operations[0],
            FusionOp::Builtin {
                origin: FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply),
                ..
            }
        ));
        let result = block_on(execute_matmul_epilogue_placed(FusionExecutionRequest {
            plan: &plan,
            inputs: inputs.clone(),
            runtime: Some(runtime.clone()),
            placement: None,
        }))
        .unwrap();
        let FusionExecutionOutcome::Executed(Value::GpuTensor(handle)) = result else {
            panic!("named override must not disable primitive numeric multiplication");
        };
        assert_eq!(
            block_on(PROVIDER.download(&handle)).unwrap().data,
            [2., 6., 4., 8.]
        );
        retire_result(handle);
        // The same represented operands through a named call retain dispatch.
        let FusionOp::Builtin { origin, .. } = &mut plan.operations[0] else {
            unreachable!()
        };
        *origin = FusionCallOrigin::named("mtimes");
        assert!(matches!(
            block_on(execute_matmul_epilogue_placed(FusionExecutionRequest {
                plan: &plan,
                inputs,
                runtime: Some(runtime),
                placement: None,
            }))
            .unwrap(),
            FusionExecutionOutcome::SharedRuntime(_)
        ));
    });
}
