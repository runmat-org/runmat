//! Exact selected constructor admission for a native invocation's bound call.

use std::rc::Rc;

use runmat_runtime::{
    call::RootedFunctionArguments, context::RuntimeContext, execution::RootedValueSequence,
    RuntimeError,
};
use runmat_types::ClassMethodOwner;
use runmat_value::{BoundFunctionBinding, LexicalEnvironment};

use crate::ExecutableUnit;

pub(super) struct Invocation {
    pub runtime: RuntimeContext,
    pub binding: BoundFunctionBinding,
    pub owner: ClassMethodOwner,
    pub product: Option<Rc<runmat_vm::SelectedClassProduct>>,
    pub registry: Rc<runmat_vm::FunctionRegistry>,
    pub unit: ExecutableUnit,
    pub executor: Rc<runmat_native_executor::NativeExecutor>,
    pub arguments: RootedFunctionArguments,
    pub requested_outputs: usize,
    pub lexical_environment: Option<LexicalEnvironment>,
    pub is_native: bool,
    pub dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
}

pub(super) async fn invoke(
    invocation: Invocation,
    super_body: bool,
) -> Result<RootedValueSequence, RuntimeError> {
    let Invocation {
        runtime,
        binding,
        owner,
        product,
        registry,
        unit,
        executor,
        arguments,
        requested_outputs,
        lexical_environment,
        is_native,
        dynamic_admission,
    } = invocation;
    let runtime = super::isolated_procedure_runtime(runtime);
    // Complete selected ancestry is required before constructor policy can
    // allocate a receiver. Unused companion definitions need not be complete.
    let runtime = if super_body {
        // The caller already selected the full Derived class product and
        // rooted its receiver. A parent body is not a second allocation.
        runtime
    } else {
        product
            .ok_or_else(|| {
                super::super::error::stage(
                    "NativeClassProduct",
                    "selected bound constructor lacks its exact class product",
                )
            })?
            .invocation_runtime_with_binding(runtime, &registry, &binding)?
    };
    let argument_count = arguments.supplied_count();
    let invocation_owner = owner.clone();
    let function = runmat_hir::FunctionId(binding.function);
    let invoked_function = binding.function;
    let invocation_binding = binding.clone();
    let make_future = move |selected_runtime: RuntimeContext| async move {
        let selected_runtime =
            selected_runtime.with_constructor_receiver_for_method(Some(&invocation_owner))?;
        let execution_sources = registry.execution_source_resolver_in(&selected_runtime);
        let selected_runtime = super::super::super_constructor::bind_selected(
            selected_runtime,
            &registry,
            function,
            &execution_sources,
        )?;
        if is_native {
            // VM semantic entry issues its own frame. Only native entry
            // needs this projection here, once for this constructor body.
            let selected_runtime = selected_runtime
                .with_admitted_constructor_frame(&invocation_owner, invocation_binding)?;
            let selected = selected_runtime.clone();
            return selected
                .scope(super::native_semantic::invoke_rooted(
                    &unit,
                    executor,
                    super::native_semantic::RootedInvocation {
                        function,
                        arguments,
                        requested_outputs,
                        runtime: selected_runtime,
                        lexical_environment,
                        dynamic_admission,
                    },
                ))
                .await;
        }
        runmat_vm::invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
            invoked_function,
            arguments,
            requested_outputs,
            &registry,
            selected_runtime,
            lexical_environment,
            dynamic_admission,
        )
        .await
    };
    if super_body {
        // This parent body already owns its receiver. Admit its original
        // concrete future directly, without a second constructor callback.
        return runmat_runtime::execution::admit_executable_sequence_with(
            &runtime.clone(),
            move || make_future(runtime),
        )?
        .await;
    }
    let invoke = runmat_runtime::dispatcher::PreparedConstructorInvoke::new(&runtime, make_future)?;
    runmat_runtime::dispatcher::call_prepared_constructor_sequence_with_argument_count_in_context(
        &runtime,
        owner,
        binding,
        argument_count,
        requested_outputs,
        invoke,
    )
    .await
}
