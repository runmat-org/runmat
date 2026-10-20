//! Issued linked-function facts for selected dynamic source admission. This
//! extends the existing execution-source resolver; it is not a name registry.

use std::{collections::HashMap, rc::Rc, sync::Arc};

use runmat_execution::FunctionBindingOrigin;
use runmat_native_codegen::aot::AotProgramManifest;
use runmat_runtime::{context::RuntimeContext, user_functions};

use super::native_error;

pub(super) struct InstalledExternalSources {
    pub(super) guard: user_functions::ExecutionSourceResolverGuard,
    /// Exact retained leases issued by the already-selected linked bodies.
    /// Kept by function ID, not display name, so retired sources still reach
    /// the bound-call liveness check instead of falling through to an ID call.
    pub(super) scoped_bindings:
        Rc<HashMap<runmat_types::FunctionId, runmat_value::BoundFunctionBinding>>,
}

pub(super) fn install(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    program: &AotProgramManifest,
    runtime: &RuntimeContext,
    functions: &[user_functions::SourceFunctionInfo],
) -> Result<InstalledExternalSources, Box<runmat_runtime::RuntimeError>> {
    let owner = runmat_runtime::source_context::SelectedProgramSourceOwner::capture(
        runtime,
        &program.source_map,
    )
    .map_err(native_error)?;
    let product = program
        .product
        .selected()
        .map(|binding| binding.fingerprint())
        .transpose()
        .map_err(native_error)?;
    let by_function = Arc::new(
        functions
            .iter()
            .map(|info| (info.function, info.clone()))
            .collect::<HashMap<_, _>>(),
    );
    let mut external = HashMap::new();
    let mut scoped_bindings = HashMap::new();
    let retained_program = Arc::new(program.clone());
    // Every linked program issues exact native leases. A selected product
    // additionally supplies the authority required to export a static link.
    for info in functions {
        let function = executor
            .function_metadata(runmat_types::FunctionId(info.function))
            .ok_or_else(|| native_error("selected source has no linked native function"))?;
        if function.source_role != info.role || function.name != info.name {
            return Err(Box::new(native_error(
                "linked function differs from its selected source declaration",
            )));
        }
        let normalized = serde_json::to_value(function).map_err(native_error)?;
        let function_bytes = serde_json::to_vec(&normalized).map_err(native_error)?;
        let mut stream = runmat_execution::DigestStream::new();
        stream.update(b"runmat-linked-native-function-v1\0");
        stream.update(&function_bytes);
        let native_function = stream.finish();
        let call_executor = Rc::clone(&executor);
        let call_program = Arc::clone(&retained_program);
        let call_function = function.id;
        let call_origin =
            FunctionBindingOrigin::from_revision(executor.program_revision(), function.portable_id)
                .map_err(native_error)?;
        let invoke: Rc<user_functions::SelectedExternalInvocation> = Rc::new(
            move |runtime, arguments, requested_outputs, lexical_environment| {
                let executor = Rc::clone(&call_executor);
                let program = Arc::clone(&call_program);
                Box::pin(async move {
                    let _guards = super::ProgramInvocationGuards::install(
                        Rc::clone(&executor),
                        &program,
                        &runtime,
                    )
                    .map_err(|error| *error)?;
                    super::invoke_native_values(
                        executor,
                        call_function,
                        arguments,
                        requested_outputs,
                        super::isolated_runtime(runtime),
                        lexical_environment,
                    )
                    .await
                })
            },
        );
        let target = user_functions::SelectedExternalSourceTarget::from_linked_function(
            runtime,
            owner.clone(),
            info,
            function.source,
            function.abi.fixed_outputs.len(),
            function.abi.varargout.is_some(),
            product,
            program.native_ir_digest,
            native_function,
            user_functions::LinkedNativeInvocation {
                compiled_function: call_function,
                origin: call_origin,
                invoke,
            },
        )
        .ok_or_else(|| native_error("linked source differs from its issued native function"))?;
        let scoped_binding = target
            .binding()
            .ok_or_else(|| native_error("selected native target has no retained binding"))?;
        if scoped_bindings
            .insert(function.id, scoped_binding)
            .is_some()
        {
            return Err(Box::new(native_error(
                "selected source repeats a native function",
            )));
        }
        if external.insert(target.origin(), target).is_some() {
            return Err(Box::new(native_error(
                "selected source repeats a native origin",
            )));
        }
    }
    let external = Rc::new(external);
    let by_function_lookup = Arc::clone(&by_function);
    let by_source_lookup = Arc::clone(&by_function);
    let by_origin_lookup = Arc::clone(&by_function);
    let by_external_lookup = Rc::clone(&external);
    let facts = user_functions::ExecutionSourceFacts::new(
        Rc::new(move |function| by_function_lookup.get(&function).cloned()),
        Rc::new(move |source| {
            by_source_lookup
                .values()
                .filter(|info| info.source_id == source)
                .cloned()
                .collect()
        }),
        Rc::new(|_| None),
        Rc::new(move |origin: FunctionBindingOrigin| {
            by_origin_lookup
                .values()
                .find(|info| info.binding_origin == Some(origin))
                .cloned()
        }),
    )
    .with_external_source(Rc::new(move |origin| {
        by_external_lookup.get(&origin).cloned()
    }));
    Ok(InstalledExternalSources {
        guard: user_functions::install_execution_source_resolver(Some(Rc::new(facts)))?,
        scoped_bindings: Rc::new(scoped_bindings),
    })
}
