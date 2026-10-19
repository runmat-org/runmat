//! A real native safepoint after deal must transfer exact outputs only once.

use super::*;
use futures::executor::block_on;
use runmat_native_executor::deopt::{DeoptimizationPolicy, FaultInjection, ResumeTarget};
use runmat_native_executor::execute::NativeInvocationStep;
use runmat_value::{IntegerStorage, Tensor};

const SOURCE: &str = "function [left, right, probe, count] = nativeDealDeopt(x, probeInput)\nglobal dealPasses\ndealPasses = dealPasses + 1;\n[left, right] = deal(x);\nprobe = sqrt(probeInput);\ncount = dealPasses;\nend\n";

#[test]
fn deal_outputs_keep_wide_aliases_and_pre_safepoint_effects_on_vm_resume() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).expect("session init");
        session.configure_runtime_context();
        let seeded = crate::execute_text_request_for_testing(
            &mut session,
            "global dealPasses; dealPasses = 0;",
        )
        .expect("initialize global pass count");
        assert!(seeded.error.is_none(), "{:?}", seeded.error);
        let unit = block_on(session.compile_executable_unit(
            crate::ExecutableSource::new("native-deal-deopt@1", "nativeDealDeopt.m", SOURCE),
            None,
        ))
        .expect("compile deal deoptimization fixture");
        let compiled = super::super::compile::compile(&unit, Some("nativeDealDeopt"))
            .expect("compile native fixture");
        let assembly = unit
            .prepare_native_compilation_for(Some("nativeDealDeopt"))
            .expect("prepare the same native product")
            .lower(runmat_native_codegen::NativeTarget::current())
            .expect("lower the same native product");
        let sqrt_start = u64::try_from(SOURCE.find("sqrt(").expect("fixture sqrt call"))
            .expect("source offset fits portable span");
        let sqrt_sites = assembly
            .functions
            .iter()
            .find(|function| function.id == compiled.entrypoint)
            .expect("selected native function")
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .filter(|instruction| {
                instruction.source.span.start <= sqrt_start
                    && sqrt_start < instruction.source.span.end
            })
            .filter_map(|instruction| instruction.safepoint)
            .collect::<Vec<_>>();
        let [safepoint] = sqrt_sites.as_slice() else {
            panic!("exact post-deal sqrt source must have one native safepoint: {sqrt_sites:?}");
        };
        assert!(
            compiled
                .safepoints
                .get(&compiled.entrypoint)
                .is_some_and(|points| points.contains(safepoint)),
            "selected source safepoint must belong to the executed product"
        );
        let expected = IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]);
        let argument =
            Value::Tensor(Tensor::new_integer(expected.clone(), vec![1, 2]).expect("wide tensor"));
        let runtime = unit
            .bind_runtime_sources(session.runtime_context().clone())
            .expect("bind exact native product sources for this invocation");
        let mut invocation = compiled
            .executor
            .begin_request(runmat_native_executor::NativeInvocationRequest {
                function: compiled.entrypoint,
                captures: Vec::new(),
                arguments: vec![argument, Value::Num(81.0)],
                requested_outputs: 4,
                runtime: runtime.clone(),
                deoptimization: DeoptimizationPolicy {
                    target: ResumeTarget::Interpreter,
                    ..DeoptimizationPolicy::default()
                }
                .inject(FaultInjection::Safepoint(*safepoint)),
                osr_target: None,
                workspace: None,
            })
            .expect("begin selected native product");
        let execution = block_on(async {
            loop {
                match invocation.advance().expect("advance native product") {
                    NativeInvocationStep::Completed(_) => {
                        panic!("selected safepoint must physically deoptimize")
                    }
                    NativeInvocationStep::Suspended {
                        continuation,
                        generation,
                    } => invocation
                        .resume_suspension(continuation, generation)
                        .await
                        .expect("resume exact native suspension"),
                    NativeInvocationStep::Deoptimized { target, frame, .. } => {
                        assert_eq!(
                            target,
                            runmat_runtime::native::NativeResumeKind::INTERPRETER
                        );
                        let environment = invocation
                            .deoptimized_lexical_environment()
                            .expect("retain exact deoptimized activation");
                        break super::resume_interpreter(
                            &unit,
                            frame,
                            4,
                            runtime.clone(),
                            environment,
                            None,
                        )
                        .await
                        .expect("resume exact VM frame");
                    }
                }
            }
        });
        let [Value::Tensor(left), Value::Tensor(right), Value::Num(probe), Value::Num(count)] =
            execution.outputs.as_slice()
        else {
            panic!("expected exact deal outputs, probe, and side-effect count");
        };
        assert_eq!(left.integer_storage(), Some(&expected));
        assert_eq!(right.integer_storage(), Some(&expected));
        assert!(left
            .host_buffer()
            .shares_allocation_with(right.host_buffer()));
        assert_eq!(*probe, 9.0);
        assert_eq!(*count, 1.0, "the pre-safepoint effect must not replay");
    });
}
