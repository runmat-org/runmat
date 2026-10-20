//! Earlier-request callees must survive producer teardown as original products.
use super::*;

#[tokio::test]
async fn original_transitive_dependencies_survive_cold_native_execution_and_redefinition() {
    let mut producer = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::default());
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
    run(
        &mut producer,
        "function y = originalLeaf(x); y = x + 100; end",
    )
    .await
    .unwrap();
    let unchanged = capture(&mut producer, &service).await;
    assert_eq!(unchanged.envelope(), original.envelope());
    drop(producer);
    let envelope = original.envelope();
    assert_eq!(
        envelope.dependency_products.len(),
        2,
        "flat original leaf and middle inventory"
    );
    let restored = ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    assert_eq!(
        restored
            .portable_envelope_for_function(original.function_in(&restored))
            .unwrap(),
        envelope
    );
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
    }
}

#[tokio::test]
async fn nested_cold_future_capture_preserves_original_dependency_product() {
    let leaf_source = "async function y = originalLeaf(x); y = x + 7; end";
    let mut producer = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::default());
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
    let cold_capture = Rc::new(CapturingService::default());
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
