#![allow(clippy::result_large_err)]

#[cfg(feature = "native-accel")]
mod tests {
    use futures::executor::block_on;
    use once_cell::sync::Lazy;
    use runmat_accelerate::{fusion_residency, simple_provider::InProcessProvider};
    use runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard};
    use runmat_builtins::Value;
    use runmat_hir::{lower, LoweringContext};
    use runmat_mir::lowering::lower_assembly;
    use runmat_runtime::RuntimeError;
    use runmat_vm::{compile_semantic_function_registry, Bytecode};

    static TEST_PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);

    fn upload_provider_handle(
        data: Vec<f64>,
        shape: Vec<usize>,
    ) -> runmat_accelerate_api::GpuTensorHandle {
        TEST_PROVIDER
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload should succeed")
    }

    fn compile_semantic_function_bytecode(
        source: &str,
        function_name: &str,
    ) -> Result<(Bytecode, usize, usize), RuntimeError> {
        let ast = runmat_parser::parse(source).map_err(|err| RuntimeError::new(err.to_string()))?;
        let hir = lower(&ast, &LoweringContext::empty())
            .map_err(|err| RuntimeError::new(err.to_string()))?;
        let mir =
            lower_assembly(&hir.assembly).map_err(|err| RuntimeError::new(format!("{err:?}")))?;

        let function_id = hir
            .assembly
            .functions
            .iter()
            .find(|function| function.name.0 == function_name)
            .map(|function| function.id)
            .ok_or_else(|| RuntimeError::new(format!("missing function `{function_name}`")))?;

        let registry =
            compile_semantic_function_registry(&hir.assembly, &mir).map_err(RuntimeError::from)?;
        let function_bytecode = registry
            .get(&function_id)
            .ok_or_else(|| RuntimeError::new("missing semantic function bytecode"))?;
        let input_slot = *function_bytecode
            .input_slots
            .first()
            .ok_or_else(|| RuntimeError::new("semantic function missing input slot"))?;
        let output_slot = *function_bytecode
            .output_slots
            .first()
            .ok_or_else(|| RuntimeError::new("semantic function missing output slot"))?;

        let mut bytecode = Bytecode::with_instructions(
            function_bytecode.instructions.clone(),
            function_bytecode.var_count,
        );
        bytecode.instr_spans = function_bytecode.instr_spans.clone();
        bytecode.call_arg_spans = function_bytecode.call_arg_spans.clone();
        bytecode.semantic_functions = registry;

        Ok((bytecode, input_slot, output_slot))
    }

    fn compile_semantic_function_invocation_fixture(
        source: &str,
        function_name: &str,
    ) -> Result<
        (
            runmat_hir::FunctionId,
            runmat_vm::SemanticFunctionRegistry,
            usize,
        ),
        RuntimeError,
    > {
        let ast = runmat_parser::parse(source).map_err(|err| RuntimeError::new(err.to_string()))?;
        let hir = lower(&ast, &LoweringContext::empty())
            .map_err(|err| RuntimeError::new(err.to_string()))?;
        let mir =
            lower_assembly(&hir.assembly).map_err(|err| RuntimeError::new(format!("{err:?}")))?;

        let function_id = hir
            .assembly
            .functions
            .iter()
            .find(|function| function.name.0 == function_name)
            .map(|function| function.id)
            .ok_or_else(|| RuntimeError::new(format!("missing function `{function_name}`")))?;

        let semantic_functions =
            compile_semantic_function_registry(&hir.assembly, &mir).map_err(RuntimeError::from)?;
        let function_bytecode = semantic_functions
            .get(&function_id)
            .ok_or_else(|| RuntimeError::new("missing semantic function bytecode"))?;
        let input_slot = *function_bytecode
            .input_slots
            .first()
            .ok_or_else(|| RuntimeError::new("semantic function missing input slot"))?;
        let registry = runmat_vm::SemanticFunctionRegistry::new(semantic_functions);
        Ok((function_id, registry, input_slot))
    }

    fn result_contains_handle(
        vars: &[Value],
        handle: &runmat_accelerate_api::GpuTensorHandle,
    ) -> bool {
        vars.iter().any(|value| {
            matches!(
                value,
                Value::GpuTensor(candidate) if candidate.buffer_id == handle.buffer_id
            )
        })
    }

    fn value_matches_handle(
        value: &Value,
        handle: &runmat_accelerate_api::GpuTensorHandle,
    ) -> bool {
        matches!(
            value,
            Value::GpuTensor(candidate) if candidate.buffer_id == handle.buffer_id
        )
    }

    #[test]
    fn semantic_spawn_overwrite_releases_unaliased_provider_handle() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![1.0, 2.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            function y = spawn_drop_unaliased(x)
                task = spawn(x);
                x = 0;
                task = 0;
                y = 0;
            end
        "#;
        let (bytecode, input_slot, _) =
            compile_semantic_function_bytecode(source, "spawn_drop_unaliased")
                .expect("compile semantic function");

        let mut vars = vec![Value::Num(0.0); bytecode.var_count];
        vars[input_slot] = Value::GpuTensor(handle.clone());

        let result = block_on(runmat_vm::interpret_function(&bytecode, vars))
            .expect("semantic spawn/drop function should run");
        assert!(
            !result_contains_handle(&result, &handle),
            "unaliased spawn/drop flow should not retain the input handle in function slots"
        );
        assert!(
            !fusion_residency::is_resident(&handle),
            "unaliased spawn/drop flow should clear residency for dropped input handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "unaliased spawn/drop flow should release provider storage for dropped input handle"
        );
    }

    #[test]
    fn semantic_spawn_overwrite_preserves_provider_handle_when_alias_retained() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![3.0, 4.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            function y = spawn_drop_with_alias(x)
                alias = x;
                task = spawn(x);
                x = 0;
                task = 0;
                y = alias;
            end
        "#;
        let (bytecode, input_slot, _) =
            compile_semantic_function_bytecode(source, "spawn_drop_with_alias")
                .expect("compile semantic function");

        let mut vars = vec![Value::Num(0.0); bytecode.var_count];
        vars[input_slot] = Value::GpuTensor(handle.clone());

        let result = block_on(runmat_vm::interpret_function(&bytecode, vars))
            .expect("semantic spawn/alias function should run");
        assert!(
            result_contains_handle(&result, &handle),
            "aliased spawn/drop flow should retain the handle when returned by alias"
        );
        assert!(
            fusion_residency::is_resident(&handle),
            "aliased spawn/drop flow should preserve residency for retained alias handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "aliased spawn/drop flow should preserve provider storage for retained alias handle"
        );

        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[test]
    fn semantic_async_spawn_await_helper_overwrite_releases_unaliased_provider_handle() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![5.0, 6.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            async function y = pass(x)
                y = x;
            end

            async function y = spawn_await_drop_unaliased(x)
                task = spawn(pass(x));
                tmp = await(task);
                task = 0;
                x = 0;
                tmp = 0;
                y = 0;
            end
        "#;
        let (function_id, registry, _input_slot) =
            compile_semantic_function_invocation_fixture(source, "spawn_await_drop_unaliased")
                .expect("compile semantic async function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[Value::GpuTensor(handle.clone())],
            1,
            &registry,
        ))
        .expect("semantic async spawn/await function should run via semantic invoker");
        assert_eq!(
            result,
            Value::Num(0.0),
            "async spawn/await helper unaliased flow should preserve scalar output semantics"
        );
        assert!(
            !fusion_residency::is_resident(&handle),
            "async spawn/await helper unaliased flow should clear residency for dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "async spawn/await helper unaliased flow should release provider storage for dropped handle"
        );
    }

    #[test]
    fn semantic_async_spawn_await_overwrite_preserves_provider_handle_when_alias_retained() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![7.0, 8.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            async function y = spawn_await_drop_with_alias(x)
                alias = x;
                task = spawn(x);
                tmp = await(task);
                task = 0;
                x = 0;
                tmp = 0;
                y = alias;
            end
        "#;
        let (bytecode, input_slot, _) =
            compile_semantic_function_bytecode(source, "spawn_await_drop_with_alias")
                .expect("compile semantic async function");

        let mut vars = vec![Value::Num(0.0); bytecode.var_count];
        vars[input_slot] = Value::GpuTensor(handle.clone());

        let result = block_on(runmat_vm::interpret_function(&bytecode, vars))
            .expect("semantic async spawn/await alias function should run");
        assert!(
            result_contains_handle(&result, &handle),
            "async spawn/await aliased flow should retain the handle when returned by alias"
        );
        assert!(
            fusion_residency::is_resident(&handle),
            "async spawn/await aliased flow should preserve residency for retained alias handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "async spawn/await aliased flow should preserve provider storage for retained alias handle"
        );

        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[test]
    fn semantic_async_spawn_await_struct_helper_releases_unaliased_provider_handle() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![9.0, 10.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            async function y = pass_struct(x)
                y = struct('payload', x);
            end

            async function y = spawn_await_drop_struct_unaliased(x)
                task = spawn(pass_struct(x));
                tmp = await(task);
                task = 0;
                x = 0;
                tmp = 0;
                y = 0;
            end
        "#;
        let (function_id, registry, _input_slot) = compile_semantic_function_invocation_fixture(
            source,
            "spawn_await_drop_struct_unaliased",
        )
        .expect("compile semantic async struct helper function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[Value::GpuTensor(handle.clone())],
            1,
            &registry,
        ))
        .expect("semantic async struct helper flow should run via semantic invoker");
        assert_eq!(
            result,
            Value::Num(0.0),
            "async spawn/await struct helper unaliased flow should preserve scalar output semantics"
        );
        assert!(
            !fusion_residency::is_resident(&handle),
            "async spawn/await struct helper unaliased flow should clear residency for dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "async spawn/await struct helper unaliased flow should release provider storage for dropped handle"
        );
    }

    #[test]
    fn semantic_async_spawn_await_cell_helper_releases_unaliased_provider_handle() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![11.0, 12.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            async function y = pass_cell(x)
                y = {x};
            end

            async function y = spawn_await_drop_cell_unaliased(x)
                task = spawn(pass_cell(x));
                tmp = await(task);
                task = 0;
                x = 0;
                tmp = 0;
                y = 0;
            end
        "#;
        let (function_id, registry, _input_slot) =
            compile_semantic_function_invocation_fixture(source, "spawn_await_drop_cell_unaliased")
                .expect("compile semantic async cell helper function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[Value::GpuTensor(handle.clone())],
            1,
            &registry,
        ))
        .expect("semantic async cell helper flow should run via semantic invoker");
        assert_eq!(
            result,
            Value::Num(0.0),
            "async spawn/await cell helper unaliased flow should preserve scalar output semantics"
        );
        assert!(
            !fusion_residency::is_resident(&handle),
            "async spawn/await cell helper unaliased flow should clear residency for dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "async spawn/await cell helper unaliased flow should release provider storage for dropped handle"
        );
    }

    #[test]
    fn semantic_async_spawn_multi_output_helper_unrequested_handle_releases() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![13.0, 14.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            async function [a,b] = pass_pair(x)
                a = 0;
                b = x;
            end

            async function y = spawn_await_drop_multi_output_unaliased(x)
                task = spawn(pass_pair(x));
                tmp = await(task);
                task = 0;
                x = 0;
                tmp = 0;
                y = 0;
            end
        "#;
        let (function_id, registry, _input_slot) = compile_semantic_function_invocation_fixture(
            source,
            "spawn_await_drop_multi_output_unaliased",
        )
        .expect("compile semantic async multi-output helper function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[Value::GpuTensor(handle.clone())],
            1,
            &registry,
        ))
        .expect("semantic async multi-output helper flow should run via semantic invoker");
        assert_eq!(
            result,
            Value::Num(0.0),
            "async spawn/await multi-output helper unaliased flow should preserve scalar output semantics"
        );
        assert!(
            !fusion_residency::is_resident(&handle),
            "async spawn/await multi-output helper unaliased flow should clear residency for dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "async spawn/await multi-output helper unaliased flow should release provider storage for dropped handle"
        );
    }

    #[test]
    fn semantic_async_spawn_varargout_helper_unrequested_handle_releases() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![15.0, 16.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            async function varargout = pass_varargout(x)
                varargout = {0, x};
            end

            async function y = spawn_await_drop_varargout_unaliased(x)
                task = spawn(pass_varargout(x));
                tmp = await(task);
                task = 0;
                x = 0;
                tmp = 0;
                y = 0;
            end
        "#;
        let (function_id, registry, _input_slot) = compile_semantic_function_invocation_fixture(
            source,
            "spawn_await_drop_varargout_unaliased",
        )
        .expect("compile semantic async varargout helper function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[Value::GpuTensor(handle.clone())],
            1,
            &registry,
        ))
        .expect("semantic async varargout helper flow should run via semantic invoker");
        assert_eq!(
            result,
            Value::Num(0.0),
            "async spawn/await varargout helper unaliased flow should preserve scalar output semantics"
        );
        assert!(
            !fusion_residency::is_resident(&handle),
            "async spawn/await varargout helper unaliased flow should clear residency for dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "async spawn/await varargout helper unaliased flow should release provider storage for dropped handle"
        );
    }

    #[test]
    fn semantic_async_spawn_varargout_nested_unrequested_handle_releases() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![17.0, 18.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let source = r#"
            async function varargout = pass_varargout_nested(x)
                varargout = {0, {x}};
            end

            async function y = spawn_await_drop_varargout_nested_unaliased(x)
                task = spawn(pass_varargout_nested(x));
                tmp = await(task);
                task = 0;
                x = 0;
                tmp = 0;
                y = 0;
            end
        "#;
        let (function_id, registry, _input_slot) = compile_semantic_function_invocation_fixture(
            source,
            "spawn_await_drop_varargout_nested_unaliased",
        )
        .expect("compile semantic async nested varargout helper function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[Value::GpuTensor(handle.clone())],
            1,
            &registry,
        ))
        .expect("semantic async nested varargout helper flow should run via semantic invoker");
        assert_eq!(
            result,
            Value::Num(0.0),
            "async spawn/await nested varargout helper unaliased flow should preserve scalar output semantics"
        );
        assert!(
            !fusion_residency::is_resident(&handle),
            "async spawn/await nested varargout helper unaliased flow should clear residency for dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "async spawn/await nested varargout helper unaliased flow should release provider storage for dropped handle"
        );
    }

    #[test]
    fn semantic_async_spawn_parallel_await_keeps_retained_handle_and_releases_dropped_handle() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle_a = upload_provider_handle(vec![21.0, 22.0], vec![1, 2]);
        let handle_b = upload_provider_handle(vec![23.0, 24.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle_a)).is_ok());
        assert!(block_on(TEST_PROVIDER.download(&handle_b)).is_ok());
        fusion_residency::mark(&handle_a);
        fusion_residency::mark(&handle_b);

        let source = r#"
            async function y = pass(x)
                y = x;
            end

            async function y = spawn_parallel_keep_first_drop_second(a, b)
                t1 = spawn(pass(a));
                t2 = spawn(pass(b));
                v2 = await(t2);
                t2 = 0;
                b = 0;
                v2 = 0;
                v1 = await(t1);
                t1 = 0;
                a = 0;
                y = v1;
            end
        "#;
        let (function_id, registry, _input_slot) = compile_semantic_function_invocation_fixture(
            source,
            "spawn_parallel_keep_first_drop_second",
        )
        .expect("compile semantic parallel async function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[
                Value::GpuTensor(handle_a.clone()),
                Value::GpuTensor(handle_b.clone()),
            ],
            1,
            &registry,
        ))
        .expect("semantic parallel async function should run via semantic invoker");
        assert!(
            value_matches_handle(&result, &handle_a),
            "parallel async flow should preserve retained first awaited handle as output"
        );
        assert!(
            fusion_residency::is_resident(&handle_a),
            "parallel async flow should preserve residency for retained first handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle_a)).is_ok(),
            "parallel async flow should preserve provider storage for retained first handle"
        );
        assert!(
            !fusion_residency::is_resident(&handle_b),
            "parallel async flow should clear residency for dropped second handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle_b)).is_err(),
            "parallel async flow should release provider storage for dropped second handle"
        );

        fusion_residency::clear(&handle_a);
        let _ = TEST_PROVIDER.free(&handle_a);
    }

    #[test]
    fn semantic_async_spawn_parallel_await_releases_both_unaliased_handles() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle_a = upload_provider_handle(vec![25.0, 26.0], vec![1, 2]);
        let handle_b = upload_provider_handle(vec![27.0, 28.0], vec![1, 2]);
        assert!(block_on(TEST_PROVIDER.download(&handle_a)).is_ok());
        assert!(block_on(TEST_PROVIDER.download(&handle_b)).is_ok());
        fusion_residency::mark(&handle_a);
        fusion_residency::mark(&handle_b);

        let source = r#"
            async function y = pass(x)
                y = x;
            end

            async function y = spawn_parallel_drop_both(a, b)
                t1 = spawn(pass(a));
                t2 = spawn(pass(b));
                v2 = await(t2);
                v1 = await(t1);
                t1 = 0;
                t2 = 0;
                a = 0;
                b = 0;
                v1 = 0;
                v2 = 0;
                y = 0;
            end
        "#;
        let (function_id, registry, _input_slot) =
            compile_semantic_function_invocation_fixture(source, "spawn_parallel_drop_both")
                .expect("compile semantic parallel async drop-both function");
        let result = block_on(runmat_vm::invoke_semantic_function_value(
            function_id.0,
            &[
                Value::GpuTensor(handle_a.clone()),
                Value::GpuTensor(handle_b.clone()),
            ],
            1,
            &registry,
        ))
        .expect("semantic parallel async drop-both function should run via semantic invoker");
        assert_eq!(
            result,
            Value::Num(0.0),
            "parallel async drop-both flow should preserve scalar output semantics"
        );
        assert!(
            !fusion_residency::is_resident(&handle_a),
            "parallel async drop-both flow should clear residency for first dropped handle"
        );
        assert!(
            !fusion_residency::is_resident(&handle_b),
            "parallel async drop-both flow should clear residency for second dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle_a)).is_err(),
            "parallel async drop-both flow should release provider storage for first dropped handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle_b)).is_err(),
            "parallel async drop-both flow should release provider storage for second dropped handle"
        );
    }
}
