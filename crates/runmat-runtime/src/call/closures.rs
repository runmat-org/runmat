use crate::call::identity::external_qualified_display_name;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::MemberAccess;
use runmat_value::{Closure, Value};

pub fn closure_value(function_name: String, captures: Vec<Value>) -> Value {
    Value::Closure(Closure {
        function_name,
        bound_function: None,
        captures,
    })
}

pub fn semantic_closure_value(
    function: runmat_types::FunctionId,
    display_name: String,
    captures: Vec<Value>,
) -> Value {
    Value::Closure(Closure {
        function_name: display_name,
        bound_function: Some(function.0),
        captures,
    })
}

pub fn caller_class_for_function(caller_function_name: Option<&str>) -> Option<String> {
    let caller_function_name = caller_function_name?;
    if let Some((class_name, method_name)) = caller_function_name.rsplit_once('.') {
        if !class_name.is_empty() && !method_name.is_empty() {
            return Some(class_name.to_string());
        }
    }
    crate::class_registry::class_names()
        .into_iter()
        .find(|class_name| {
            crate::class_registry::get_class(class_name).is_some_and(|class_def| {
                class_def
                    .methods
                    .values()
                    .any(|method| method.function_name == caller_function_name)
            })
        })
}

pub fn method_access_permitted(
    owner: &str,
    access: &MemberAccess,
    caller_function_name: Option<&str>,
) -> bool {
    match access {
        MemberAccess::Public => true,
        MemberAccess::Private => {
            caller_class_for_function(caller_function_name).as_deref() == Some(owner)
        }
        MemberAccess::Protected => {
            caller_class_for_function(caller_function_name).is_some_and(|caller_class| {
                crate::class_registry::is_class_or_subclass(&caller_class, owner)
            })
        }
    }
}

pub fn resolve_method_semantic_function_id(
    owner: &str,
    method_name: &str,
    function_name: &str,
) -> Option<usize> {
    let trimmed = function_name.trim();
    if !trimmed.is_empty() {
        if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(trimmed) {
            return Some(function);
        }
        if !trimmed.contains('.') {
            let owner_qualified = format!("{owner}.{trimmed}");
            if let Some(function) =
                crate::user_functions::resolve_semantic_function_by_name(&owner_qualified)
            {
                return Some(function);
            }
        }
    }
    crate::user_functions::resolve_semantic_function_by_name(&format!("{owner}.{method_name}"))
}

pub fn load_method_closure(
    base: Value,
    name: String,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Object(object) => {
            let function_name = external_qualified_display_name(&object.class_name, &name);
            Ok(Value::Closure(Closure {
                bound_function: crate::user_functions::resolve_semantic_function_by_name(
                    &function_name,
                ),
                function_name,
                captures: vec![Value::Object(object)],
            }))
        }
        Value::ClassRef(class_name) => {
            if let Some((method, owner)) = crate::class_registry::lookup_method(&class_name, &name)
            {
                if !method.is_static {
                    return Err(semantic_error(
                        "MethodNotStatic",
                        format!("Method '{name}' is not static"),
                    ));
                }
                if !method_access_permitted(&owner, &method.access, caller_function_name) {
                    return Err(semantic_error(
                        "MethodPrivate",
                        format!("Method '{name}' is private"),
                    ));
                }
                return Ok(Value::Closure(Closure {
                    bound_function: resolve_method_semantic_function_id(
                        &owner,
                        &name,
                        &method.function_name,
                    ),
                    function_name: method.function_name,
                    captures: vec![],
                }));
            }
            let qualified_name = external_qualified_display_name(&class_name, &name);
            if runmat_builtins::builtin_name_is_known(&qualified_name) {
                Ok(Value::Closure(Closure {
                    bound_function: crate::user_functions::resolve_semantic_function_by_name(
                        &qualified_name,
                    ),
                    function_name: qualified_name,
                    captures: vec![],
                }))
            } else {
                Err(semantic_error(
                    "UnknownStaticMethod",
                    format!("Unknown static method '{name}' on class {class_name}"),
                ))
            }
        }
        _ => Err(semantic_error(
            "LoadMethod",
            "LoadMethod requires object or classref",
        )),
    }
}
