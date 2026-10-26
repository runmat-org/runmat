//! Array-creation methods use the prototype's original class product. A
//! selected method's error is final; only declaration absence permits fallback.
use crate::{call::BuiltinArguments, context::RuntimeContext, execution::RootedValueSequence};
use runmat_types::StaticMethodName;

const ZEROS_LIKE: StaticMethodName = StaticMethodName::new("zerosLike");
const ZEROS: StaticMethodName = StaticMethodName::new("zeros");

pub(super) fn like_prototype(args: &BuiltinArguments) -> crate::BuiltinResult<Option<usize>> {
    // The class ABI is a trailing control pair. Numeric construction keeps
    // its existing validation for other forms, including malformed controls.
    let Some(index) = args.len().checked_sub(2) else {
        return Ok(None);
    };
    if crate::builtins::common::arg_tokens::scalar_keyword_of(&args[index], "zeros")?.as_deref()
        != Some("like")
    {
        return Ok(None);
    }
    Ok(crate::object::indexing::class_name_from_base(&args[index + 1]).map(|_| index))
}

pub(super) async fn invoke(
    runtime: Option<&RuntimeContext>,
    args: BuiltinArguments,
    like_index: usize,
) -> crate::BuiltinResult<RootedValueSequence> {
    use crate::object::protocol::{
        invoke_resolved_object_method_retained, resolve_declared_object_method,
        resolve_declared_object_static_method, ObjectAccessContext,
    };
    let prototype = &args[like_index + 1];
    let access = ObjectAccessContext::default();
    let (method, arguments) = if let Some(method) =
        resolve_declared_object_method(prototype, &ZEROS_LIKE.owned(), &access)?
    {
        let count = like_index + 1;
        let arguments = args.try_map_preserving_roots(count, runtime, |slot, source| {
            Ok(source[if slot == 0 { like_index + 1 } else { slot - 1 }].clone())
        })?;
        (method, arguments)
    } else if let Some(method) =
        resolve_declared_object_static_method(prototype, &ZEROS.owned(), &access)?
    {
        (method, args.into_range(..like_index)?)
    } else {
        return Err(super::builtin_error(
            "zeros: prototype class does not define zerosLike or static zeros",
        ));
    };
    invoke_resolved_object_method_retained(runtime, &method, arguments, 1).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_registry::{RuntimeClass, RuntimeMethod};
    use futures::executor::block_on;
    use runmat_types::{ClassIdentity, ClassMethodRole, MemberAccess};
    use runmat_value::{ObjectInstance, Value};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[test]
    fn original_prototype_methods_preserve_arguments_and_do_not_replay_errors() {
        runmat_gc::gc_test_context(|| {
            let runtime = RuntimeContext::new(std::rc::Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ));
            let setup = runtime.enter();
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = calls.clone();
            let original_scope = runtime.execution().scope_id();
            let _resolver =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    match name {
                        "PrototypeCreation.instance" => Some(1),
                        "PrototypeCreation.static" => Some(2),
                        "PrototypeCreation.error" => Some(3),
                        _ => None,
                    }
                })))
                .unwrap();
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(move |function, arguments, requested| {
                    observed.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(
                        crate::context::legacy::active()
                            .unwrap()
                            .execution()
                            .scope_id(),
                        original_scope,
                    );
                    assert_eq!(requested, 1);
                    assert_eq!(arguments.len(), if function == 2 { 2 } else { 3 });
                    let offset = usize::from(function != 2);
                    assert_eq!(arguments[offset], Value::Num(2.0));
                    assert_eq!(arguments[offset + 1], Value::Num(3.0));
                    if offset == 1 {
                        assert!(matches!(arguments[0], Value::Object(_)));
                    }
                    Box::pin(async move {
                        if function == 3 {
                            return Err(crate::runtime_error::semantic_error(
                                "PrototypeCreationOrigin",
                                "original creation error",
                            ));
                        }
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(function as f64),
                        )
                    })
                }),
            ))
            .unwrap();
            let mut inputs = Vec::new();
            for (class, instance) in [
                ("PrototypeCreationBoth", Some("PrototypeCreation.instance")),
                ("PrototypeCreationStatic", None),
                ("PrototypeCreationError", Some("PrototypeCreation.error")),
            ] {
                let class = ClassIdentity::from(class);
                let mut methods = std::collections::HashMap::new();
                for (name, is_static, function) in
                    std::iter::once((ZEROS, true, "PrototypeCreation.static"))
                        .chain(instance.map(|function| (ZEROS_LIKE, false, function)))
                {
                    methods.insert(
                        name.owned(),
                        RuntimeMethod {
                            name: name.owned(),
                            is_static,
                            is_abstract: false,
                            is_sealed: false,
                            access: MemberAccess::Public,
                            function_name: function.into(),
                            implicit_class_argument: None,
                            role: ClassMethodRole::Ordinary,
                        },
                    );
                }
                crate::class_registry::register_class(RuntimeClass {
                    name: class.clone(),
                    parent: None,
                    properties: Default::default(),
                    events: Default::default(),
                    methods,
                });
                inputs.push(
                    BuiltinArguments::from_borrowed(
                        &[
                            Value::Num(2.0),
                            Value::Num(3.0),
                            Value::from("like"),
                            Value::Object(ObjectInstance::new(class)),
                        ],
                        Some(&runtime),
                    )
                    .unwrap(),
                );
            }
            drop(setup);
            let foreign = RuntimeContext::new(std::rc::Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ));
            let _foreign = foreign.enter();
            for (index, arguments) in inputs.into_iter().enumerate() {
                let result = block_on(super::super::zeros_dispatch(
                    Some(runtime.clone()),
                    arguments,
                ));
                if index == 2 {
                    assert_eq!(
                        result.unwrap_err().identifier(),
                        Some("RunMat:PrototypeCreationOrigin")
                    );
                } else {
                    assert_eq!(
                        result.unwrap().value().as_slice(),
                        &[Value::Num((index + 1) as f64)],
                    );
                }
                assert_eq!(calls.load(Ordering::SeqCst), index + 1);
                assert_eq!(
                    crate::context::legacy::active()
                        .unwrap()
                        .execution()
                        .scope_id(),
                    foreign.execution().scope_id()
                );
            }
        });
    }
}
