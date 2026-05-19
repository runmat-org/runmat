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
}
