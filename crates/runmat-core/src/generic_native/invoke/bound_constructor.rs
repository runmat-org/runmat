//! Exact selected constructor admission for a native invocation's bound call.

use std::{collections::BTreeMap, rc::Rc, sync::Arc};

use runmat_runtime::{
    context::RuntimeContext,
    execution::{RootedValueSequence, RootedValues},
    user_functions, RuntimeError,
};
use runmat_types::{BindingId, ClassMethodOwner, ProgramFunctionId};
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
    pub capture_bindings: Arc<BTreeMap<ProgramFunctionId, Vec<BindingId>>>,
    pub direct: Option<Rc<user_functions::FunctionInvoker>>,
    pub arguments: RootedValues,
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
        capture_bindings,
        direct,
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
            .invocation_runtime(runtime)?
    };
    let policy_arguments = Rc::new(arguments);
    let invoke_arguments = Rc::clone(&policy_arguments);
    let invocation_owner = owner.clone();
    let function = runmat_hir::FunctionId(binding.function);
    let invoked_function = binding.function;
    let invoke: runmat_runtime::dispatcher::PreparedConstructorInvoke =
        Box::new(move |selected_runtime| {
            Box::pin(async move {
                let selected_runtime = selected_runtime
                    .with_constructor_receiver_for_method(Some(&invocation_owner))?;
                let execution_sources = registry.execution_source_resolver_in(&selected_runtime);
                let selected_runtime = super::super::super_constructor::bind_selected(
                    selected_runtime,
                    &registry,
                    function,
                    &execution_sources,
                )?;
                if is_native && lexical_environment.is_some() {
                    let function = super::program_function(function)?;
                    let selected = selected_runtime.clone();
                    return selected
                        .scope(super::native_semantic::invoke(
                            &unit,
                            super::super::deopt::DeoptimizationRequest {
                                executor,
                                function,
                                captures: Vec::new(),
                                lexical_environment,
                                arguments: invoke_arguments.values().to_vec(),
                                requested_outputs,
                                runtime: selected_runtime,
                                dynamic_admission,
                                osr: None,
                                workspace: None,
                            },
                            capture_bindings
                                .get(&function)
                                .map(Vec::as_slice)
                                .unwrap_or(&[]),
                        ))
                        .await;
                }
                if let Some(direct) = direct.filter(|_| lexical_environment.is_none()) {
                    return selected_runtime
                        .scope(direct(
                            Some(selected_runtime.clone()),
                            invoked_function,
                            invoke_arguments.values(),
                            requested_outputs,
                        ))
                        .await;
                }
                if let Some(dynamic_admission) = dynamic_admission {
                    runmat_vm::invoke_semantic_function_value_with_dynamic_admission_in_context(
                        invoked_function,
                        invoke_arguments.values(),
                        requested_outputs,
                        &registry,
                        selected_runtime,
                        lexical_environment,
                        dynamic_admission,
                    )
                    .await
                } else {
                    runmat_vm::invoke_semantic_function_value_with_environment_in_context(
                        invoked_function,
                        invoke_arguments.values(),
                        requested_outputs,
                        &registry,
                        selected_runtime,
                        lexical_environment,
                    )
                    .await
                }
            })
        });
    if super_body {
        return invoke(runtime).await;
    }
    runmat_runtime::dispatcher::call_prepared_constructor_sequence_in_context(
        &runtime,
        owner,
        binding,
        policy_arguments.values(),
        requested_outputs,
        invoke,
    )
    .await
}
