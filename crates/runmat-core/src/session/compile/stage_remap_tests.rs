#[test]
fn compiled_local_helper_and_callback_keep_selected_stage_facts_in_v2_product() {
    let source = crate::ExecutableSource::new(
        "fixture",
        "main.m",
        "a=helper(1); b=feval('helper',2); out=a+b; function y=helper(x); mu=x+1; sigma=mu.*2; y=sigma-3; end",
    );
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let unit = futures::executor::block_on(session.compile_executable_unit(source, None))
        .expect("compiled unit");
    let envelope = unit.portable_envelope().expect("V2 envelope");
    let bytes = envelope.canonical_bytes().expect("canonical V2 bytes");
    let admitted = runmat_vm::admit_executable_unit(&bytes).expect("selected V2 admission");
    assert!(admitted.registry().functions.values().any(|function| {
        function.display_name == "helper" && function.fusion_stage_projection.is_some()
    }));
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let runtime = unit
        .bind_runtime_sources(runtime)
        .expect("selected source leases");
    let executable = unit
        .runtime_bytecode(&runtime)
        .expect("runtime source projection");
    assert!(executable.bound_functions.values().any(|function| {
        function.display_name == "helper" && function.fusion_stage_projection.is_some()
    }));
}
