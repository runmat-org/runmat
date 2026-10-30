//! Earlier-request callees must survive producer teardown as original products.
use super::*;

#[tokio::test]
async fn original_transitive_dependencies_survive_cold_native_execution_and_redefinition() {
    let mut producer = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::new(&producer));
    producer.install_execution_services(service.clone());
    run(
        &mut producer,
        "function y = originalLeaf(x); y = x + 7; end",
    )
    .await
    .unwrap();
    run(
        &mut producer,
        "function y = originalMiddle(x); y = originalLeaf(x) * 2; end",
    )
    .await
    .unwrap();
    run(
        &mut producer,
        "async function y = retainedAsync(x); y = originalMiddle(x) + 3; end",
    )
    .await
    .unwrap();
    let original = capture(&mut producer, &service).await;
    assert!(
        original.request.static_dependencies.len() >= 2,
        "the actual Graph request must retain the transitive selected static links"
    );
    let static_origins = original
        .request
        .static_dependencies
        .iter()
        .map(|edge| edge.target.origin)
        .collect::<std::collections::HashSet<_>>();
    run(
        &mut producer,
        "function y = originalLeaf(x); y = x + 100; end",
    )
    .await
    .unwrap();
    let unchanged = capture(&mut producer, &service).await;
    assert_eq!(
        unchanged
            .request
            .static_dependencies
            .iter()
            .map(|edge| edge.target.origin)
            .collect::<std::collections::HashSet<_>>(),
        static_origins,
        "later name publication must not change the Graph's original static targets"
    );
    assert_eq!(unchanged.envelope(), original.envelope());
    let portable_capture = {
        let runmat_runtime::execution::DeferredProgram::SelectedGraph(program) =
            &original.call.program
        else {
            panic!("original graph call must retain its real selected producer");
        };
        let runmat_runtime::execution::DeferredInvocation::Callable(mut descriptor) =
            original.call.invocation.clone()
        else {
            panic!("original graph must have a callable target");
        };
        descriptor.args.clear();
        let selected = [*descriptor];
        program
            .capture_callables(&selected, runmat_execution::value::ValueLimits::default())
            .expect("actual selected producer captures the original callback graph")
            .into_portable()
    };
    let portable_capture: runmat_execution::value::PortableSelectedCode =
        serde_json::from_slice(&serde_json::to_vec(&portable_capture).unwrap()).unwrap();
    assert!(portable_capture.selected_foreign_policies.is_empty());
    drop(producer);
    let envelope = original.envelope();
    assert_eq!(
        envelope.dependency_products.len(),
        2,
        "flat original leaf and middle inventory"
    );
    let static_section = |manifest: &runmat_execution::ExecutableUnitManifest| {
        manifest
            .optional_sections
            .iter()
            .position(|section| section.name == runmat_execution::executable::STATIC_IMPORT_SECTION)
    };
    assert!(
        static_section(&envelope.manifest).is_some(),
        "standalone caller preserves its original selected static imports"
    );
    assert!(
        envelope
            .dependency_products
            .iter()
            .any(|product| { static_section(&product.manifest).is_some() }),
        "the original middle product preserves its own outbound leaf import"
    );
    let restored = ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    assert_eq!(
        restored
            .portable_envelope_for_function(original.function_in(&restored))
            .unwrap(),
        envelope
    );
    for mutation in 0..3 {
        let mut changed = envelope.clone();
        let position = static_section(&changed.manifest).unwrap();
        if mutation == 0 {
            changed.manifest.optional_sections.remove(position);
        } else {
            let mut imports = runmat_execution::executable::decode_static_imports(
                &changed.manifest.optional_sections[position],
            )
            .unwrap();
            if mutation == 1 {
                imports[0].target_content = runmat_execution::executable::SelectedUnitContentDigest(
                    runmat_execution::Digest::sha256(b"substituted original content"),
                );
            } else {
                imports[0].target_body = runmat_execution::executable::SelectedBodySemanticDigest(
                    runmat_execution::Digest::sha256(b"substituted original body"),
                );
            }
            changed.manifest.optional_sections[position] =
                runmat_execution::executable::static_import_section(imports).unwrap();
        }
        changed.validate().unwrap();
        assert!(
            ExecutableUnit::from_portable_envelope(&changed).is_err(),
            "original dependency inventory does not authorize missing or substituted site seals"
        );
    }
    let incomplete = runmat_execution::ExecutableUnitEnvelope::new(
        envelope.manifest.clone(),
        envelope.payloads.clone(),
    )
    .unwrap();
    assert!(
        ExecutableUnit::from_portable_envelope(&incomplete).is_err(),
        "copied bytecode does not replace missing original compiler products"
    );
    let mut changed = envelope.clone();
    let kind = runmat_execution::ExecutableComponentKind::FunctionRegistry;
    let payload = changed
        .payloads
        .iter_mut()
        .find(|payload| payload.kind == kind)
        .unwrap();
    let mut registry: runmat_vm::FunctionRegistry = serde_json::from_slice(&payload.bytes).unwrap();
    registry
        .functions
        .values_mut()
        .find(|function| function.display_name == "originalMiddle")
        .unwrap()
        .private_owner_scope = "substituted".into();
    payload.bytes = serde_json::to_vec(&serde_json::to_value(&registry).unwrap()).unwrap();
    let descriptor = changed
        .manifest
        .components
        .iter_mut()
        .find(|descriptor| descriptor.kind == kind)
        .unwrap();
    descriptor.encoded_length = payload.bytes.len() as u64;
    descriptor.digest = runmat_execution::Digest::sha256(&payload.bytes);
    assert!(
        changed.validate().is_ok(),
        "transport hashes alone are internally consistent"
    );
    assert!(
        ExecutableUnit::from_portable_envelope(&changed).is_err(),
        "foreign execution projection must match original owner"
    );
    for native in [false, true] {
        let mut cold = RunMatSession::with_options(false, false).unwrap();
        cold.set_compat_mode(runmat_parser::CompatMode::RunMat);
        let mut control = crate::InvocationControl::default();
        if native {
            control = control.force_generic_native();
        }
        let result = cold
            .invoke_portable_unit(
                &envelope,
                original.call.invocation.arguments().to_vec(),
                original.call.invocation.requested_outputs(),
                &control,
            )
            .await
            .expect("original transitive product must execute without producer registry");
        assert_eq!(result.as_slice(), &[Value::Num(27.0)]);

        // Restoring code issues current-session sources without retaining the
        // admission call's cancellation generation. Later calls use B, not A.
        let admitted = runmat_execution_artifact::CapturedGraphCode::new(portable_capture.clone())
            .into_validated()
            .unwrap();
        let (admitted, bundles) = admitted.into_admitted_with_bundles();
        let old = runmat_execution::cancellation::CancellationSource::new();
        let admission_runtime = cold
            .runtime_context
            .clone()
            .with_cancellation_token(old.token());
        let (code, roots) = runmat_vm::RestoredSelectedGraph::admit_and_decode(
            &admitted,
            bundles,
            &admission_runtime,
            None,
        )
        .unwrap();
        old.cancel(runmat_execution::CancellationReason::User);
        let current = runmat_execution::cancellation::CancellationSource::new();
        let runtime = cold
            .runtime_context
            .clone()
            .with_cancellation_token(current.token());
        let callback = crate::callable_selection::select(
            &runtime,
            code.primary_registry(),
            runmat_runtime::execution::RootedValue::in_context(roots.values()[0].clone(), &runtime)
                .unwrap(),
        )
        .unwrap();
        let receipt = code
            .admit_callable(&runtime, callback.selected_descriptor())
            .unwrap()
            .unwrap();
        let arguments =
            runmat_runtime::execution::RootedValues::in_context(vec![Value::Num(5.0)], &runtime)
                .unwrap();
        let descriptor = callback
            .selected_descriptor()
            .clone()
            .with_retained_arguments(arguments.into())
            .unwrap();
        // A previous caller port must not replace the restored code's own
        // exact invoker while its returned child is actually polled.
        let calls = Rc::new(std::cell::Cell::new(0usize));
        let observed = Rc::clone(&calls);
        let active = runtime.enter();
        let previous = runmat_runtime::user_functions::install_local_bound_function_invoker(
            Rc::new(move |current, _, _, _, _| {
                observed.set(observed.get() + 1);
                runmat_runtime::execution::admit_executable_sequence_with(&current, || async {
                    Err(runmat_runtime::build_runtime_error(
                        "wrong previous restored callback invoker",
                    )
                    .build())
                })
            }),
        )
        .unwrap();
        let result = code
            .invoke(runtime.clone(), receipt.clone(), descriptor)
            .unwrap()
            .await
            .unwrap();
        assert_eq!(result.as_slice(), &[Value::Num(27.0)]);
        assert_eq!(
            calls.get(),
            0,
            "restored invocation must use its retained exact code owner"
        );
        drop(previous);
        drop(active);
        let arguments =
            runmat_runtime::execution::RootedValues::in_context(vec![Value::Num(5.0)], &runtime)
                .unwrap();
        let descriptor = callback
            .selected_descriptor()
            .clone()
            .with_retained_arguments(arguments.into())
            .unwrap();
        let pending = code.invoke(runtime, receipt, descriptor).unwrap();
        current.cancel(runmat_execution::CancellationReason::User);
        let failure = pending.await.unwrap_err();
        assert!(runmat_runtime::runtime_error::error_has_control_cancellation_origin(&failure));
    }
}

#[tokio::test]
async fn nested_cold_future_capture_preserves_original_dependency_product() {
    let leaf_source = "async function y = originalLeaf(x); y = x + 7; end";
    let mut producer = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::new(&producer));
    producer.install_execution_services(service.clone());
    run(&mut producer, leaf_source).await.unwrap();
    run(
        &mut producer,
        "async function y = retainedAsync(x); y = originalLeaf(x); end",
    )
    .await
    .unwrap();
    let original = capture(&mut producer, &service).await;
    drop(producer);
    let envelope = original.envelope();
    let mut cold = RunMatSession::with_options(false, false).unwrap();
    cold.set_compat_mode(runmat_parser::CompatMode::RunMat);
    let cold_capture = Rc::new(CapturingService::new(&cold));
    cold.install_execution_services(cold_capture.clone());
    let value = cold
        .invoke_portable_unit(
            &envelope,
            original.call.invocation.arguments().to_vec(),
            original.call.invocation.requested_outputs(),
            &crate::InvocationControl::default(),
        )
        .await
        .expect("nested capture needs the original leaf compiler authority");
    assert!(matches!(value.as_slice(), [Value::Future(_)]));
    let leaf = cold_capture.take_original();
    let captured = leaf.envelope();
    assert_eq!(
        ExecutableUnit::from_portable_envelope(&captured)
            .unwrap()
            .source()
            .text,
        leaf_source
    );
    assert_eq!(cold_value(&leaf).await.as_slice(), &[Value::Num(12.0)]);
}
