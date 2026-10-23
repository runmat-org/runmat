//! Re-admit a current class source without invoking its constructor.

use std::ops::ControlFlow;
use std::rc::Rc;

use runmat_runtime::context::{CurrentClassProductResolution, RuntimeContext};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_types::{ClassIdentity, ClassMethodRole};
use runmat_value::{BoundFunctionBinding, Value};

use super::DynamicFunctionEnvironment;

pub(in crate::session) async fn resolve(
    runtime: RuntimeContext,
    class: ClassIdentity,
    environment: Rc<DynamicFunctionEnvironment>,
) -> Result<CurrentClassProductResolution, RuntimeError> {
    let arguments = runmat_runtime::call::RootedFunctionArguments::ordinary(
        runmat_runtime::execution::RootedValues::new(Vec::new())
            .map_err(|error| error.into_runtime_error())?,
    );
    let selection = super::load_dynamic_function(
        runtime.clone(),
        class.display_name().to_owned(),
        arguments,
        1,
        runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle {
            class: Some(class.clone()),
            selected_source: None,
        },
        None,
        environment.clone(),
    )
    .await;
    let selected = match selection {
        ControlFlow::Break(Ok(selected)) => selected,
        ControlFlow::Break(Err(error)) => return Err(error),
        ControlFlow::Continue(_arguments) => {
            if let Some(issuance) =
                runmat_runtime::class_registry::issue_builtin_class_in_context(&runtime, &class)
            {
                return Ok(CurrentClassProductResolution::RegisteredBuiltin(issuance));
            }
            return Err(unavailable("current class source is unavailable"));
        }
    };
    let [Value::BoundFunctionHandle(handle)] = selected.value().as_slice() else {
        return Err(unavailable(
            "current class source did not yield a bound constructor",
        ));
    };
    let binding = BoundFunctionBinding {
        function: handle.function,
        authority: handle.authority.clone(),
    };
    let view = environment
        .source_evaluator
        .selected_binding_view(&binding)?
        .ok_or_else(|| unavailable("current constructor definition is unavailable"))?;
    let lease = binding
        .authority
        .local_lease(binding.function)
        .ok_or_else(|| unavailable("current constructor lease is unavailable"))?;
    let body = view
        .get_owned(runmat_hir::FunctionId(binding.function), lease)
        .ok_or_else(|| unavailable("current constructor body is unavailable"))?;
    let owner = body
        .class_method_owner
        .as_ref()
        .ok_or_else(|| unavailable("current constructor has no class declaration"))?;
    if owner.role != ClassMethodRole::Constructor
        || owner.is_static
        || owner.declaring_class != class
        || body
            .selected_source_owner
            .as_ref()
            .is_none_or(|source| !source.is_issued_in(&runtime))
    {
        return Err(unavailable(
            "current constructor differs from its selected class",
        ));
    }
    let product = body
        .selected_class_product
        .as_ref()
        .ok_or_else(|| unavailable("current constructor lacks its class product"))?;
    if product.class() != &class {
        return Err(unavailable(
            "current class product differs from its constructor",
        ));
    }
    let facts = product.receiver_facts_with_binding(&view, &binding)?;
    Ok(CurrentClassProductResolution::Selected(facts))
}

fn unavailable(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:ClassProduct:Unavailable")
        .build()
}
