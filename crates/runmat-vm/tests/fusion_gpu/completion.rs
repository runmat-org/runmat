//! Real semantic-return projection, distinct from low-level full-frame adoption.
use super::*;

#[test]
fn semantic_projection_releases_temporaries_on_success_and_output_error() {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        for fails in [false, true] {
            let source = if fails {
                "function varargout = projected(x)\ndiscarded = sin(x);\nvarargout = 3;\nend"
            } else {
                "function y = projected(x)\ndiscarded = sin(x);\ny = x;\nend"
            };
            let ast = runmat_parser::parse(source).unwrap();
            let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
            let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
            let registry = runmat_vm::FunctionRegistry::new(
                runmat_vm::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
            );
            let function = registry.resolve_name("projected").unwrap();
            let input = provider.push(vec![0.25, 0.5], vec![1, 2]);
            let caller = runmat_gc::gc_register_value_root(Box::new(runmat_gc::GlobalRoot::new(
                vec![Value::GpuTensor(input.clone())],
                "independent semantic caller".into(),
            )))
            .unwrap();
            let baseline = provider
                .buffers
                .lock()
                .unwrap()
                .keys()
                .copied()
                .collect::<std::collections::HashSet<_>>();
            let first = provider.next_id.load(Ordering::Relaxed);
            let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let result = block_on(runmat_vm::invoke_semantic_function_value_in_context(
                function.0,
                &[Value::GpuTensor(input.clone())],
                1,
                &registry,
                runtime,
            ));
            if fails {
                assert!(result.unwrap_err().to_string().contains("varargout"));
            } else {
                let outputs = result.unwrap();
                let [Value::GpuTensor(output)] = outputs.as_slice() else {
                    panic!("resident alias output")
                };
                assert_eq!(output.buffer_id, input.buffer_id);
            }
            let end = provider.next_id.load(Ordering::Relaxed);
            assert!(
                end > first,
                "actual semantic body must allocate a discarded result"
            );
            for id in first..end {
                assert_eq!(provider.frees.lock().unwrap().get(&id), Some(&1));
            }
            assert_eq!(
                provider
                    .buffers
                    .lock()
                    .unwrap()
                    .keys()
                    .copied()
                    .collect::<std::collections::HashSet<_>>(),
                baseline
            );
            assert_eq!(
                provider.pull(&input).unwrap(),
                (vec![0.25, 0.5], vec![1, 2])
            );
            runmat_gc::gc_unregister_root(caller).unwrap();
            provider.free(&input).unwrap();
        }
    });
}

#[test]
fn completion_projection_cancellation_retires_owned_values_but_not_published_aliases() {
    use std::future::Future;
    use std::task::{Context, Poll};
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        let owned = provider.push(vec![1.0], vec![1, 1]);
        let published = provider.push(vec![2.0], vec![1, 1]);
        let caller = runmat_gc::gc_register_value_root(Box::new(runmat_gc::GlobalRoot::new(
            vec![Value::GpuTensor(published.clone())],
            "published projection alias".into(),
        )))
        .unwrap();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let frame = vec![
            Value::GpuTensor(owned.clone()),
            Value::GpuTensor(published.clone()),
        ];
        let mut pending = Box::pin(async move {
            let _projection = runmat_vm::CompletedFrameProjection::new(frame, runtime);
            futures::future::pending::<()>().await;
        });
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        drop(pending);
        assert!(provider.pull(&owned).is_err());
        assert_eq!(
            provider.frees.lock().unwrap().get(&owned.buffer_id),
            Some(&1)
        );
        assert_eq!(provider.pull(&published).unwrap(), (vec![2.0], vec![1, 1]));
        runmat_gc::gc_unregister_root(caller).unwrap();
        provider.free(&published).unwrap();
    });
}
