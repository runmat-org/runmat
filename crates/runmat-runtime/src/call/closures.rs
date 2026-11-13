use crate::call::identity::external_qualified_display_name;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{ClassIdentity, MemberAccess};
use runmat_value::{Closure, Value};

pub fn closure_value(function_name: String, captures: Vec<Value>) -> Value {
    Value::Closure(Closure::new(
        function_name,
        None,
        runmat_types::FunctionSourceRole::Unknown,
        None,
        captures,
        None,
    ))
}

pub fn semantic_closure_value(
    function: runmat_types::FunctionId,
    display_name: String,
    captures: Vec<Value>,
    source_role: runmat_types::FunctionSourceRole,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Value {
    Value::Closure(Closure::new(
        display_name,
        Some(crate::user_functions::bound_function_binding_for(
            function.0,
        )),
        source_role,
        crate::user_functions::source_file_for(function.0),
        captures,
        lexical_environment,
    ))
}

pub fn caller_class_for_function(caller_function_name: Option<&str>) -> Option<ClassIdentity> {
    caller_function_name
        .and_then(crate::class_registry::caller_method_for_function)
        .map(|(class, _)| class)
}

pub fn method_access_permitted(
    owner: &ClassIdentity,
    access: &MemberAccess,
    caller_function_name: Option<&str>,
) -> bool {
    match access {
        MemberAccess::Public => true,
        MemberAccess::Private => {
            caller_class_for_function(caller_function_name).as_ref() == Some(owner)
        }
        MemberAccess::Protected => {
            caller_class_for_function(caller_function_name).is_some_and(|caller_class| {
                crate::class_registry::is_class_or_subclass(&caller_class, owner)
            })
        }
    }
}

pub fn method_access_permitted_for_class(
    owner: &ClassIdentity,
    access: &MemberAccess,
    caller_class: Option<&ClassIdentity>,
) -> bool {
    match access {
        MemberAccess::Public => true,
        MemberAccess::Private => caller_class == Some(owner),
        MemberAccess::Protected => caller_class
            .is_some_and(|caller| crate::class_registry::is_class_or_subclass(caller, owner)),
    }
}

pub fn resolve_method_semantic_function_id(
    owner: &ClassIdentity,
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
            let function_name =
                external_qualified_display_name(object.class_name().display_name(), &name);
            let registered = crate::class_registry::lookup_bound_method(
                object.class_name(),
                &runmat_types::MethodName::from(name.as_str()),
            )
            .and_then(|method| method.bound_function);
            let bound_function =
                registered
                    .as_ref()
                    .map(|binding| binding.function)
                    .or_else(|| {
                        crate::user_functions::resolve_semantic_function_by_name(&function_name)
                    });
            Ok(Value::Closure(Closure::new(
                function_name,
                registered.or_else(|| {
                    bound_function.map(crate::user_functions::bound_function_binding_for)
                }),
                runmat_types::FunctionSourceRole::Method,
                bound_function.and_then(crate::user_functions::source_file_for),
                vec![Value::Object(object)],
                None,
            )))
        }
        Value::ClassRef(class_name) => {
            let method_name = runmat_types::MethodName::from(name.as_str());
            if let Some((method, owner)) =
                crate::class_registry::lookup_method(&class_name, &method_name)
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
                let registered =
                    crate::class_registry::lookup_bound_method(&class_name, &method_name)
                        .and_then(|method| method.bound_function);
                let bound_function =
                    registered
                        .as_ref()
                        .map(|binding| binding.function)
                        .or_else(|| {
                            resolve_method_semantic_function_id(
                                &owner,
                                &name,
                                &method.function_name,
                            )
                        });
                return Ok(Value::Closure(Closure::new(
                    method.function_name,
                    registered.or_else(|| {
                        bound_function.map(crate::user_functions::bound_function_binding_for)
                    }),
                    runmat_types::FunctionSourceRole::Method,
                    bound_function.and_then(crate::user_functions::source_file_for),
                    vec![],
                    None,
                )));
            }
            let qualified_name = external_qualified_display_name(class_name.display_name(), &name);
            if runmat_builtins::builtin_name_is_known(&qualified_name) {
                let bound_function =
                    crate::user_functions::resolve_semantic_function_by_name(&qualified_name);
                Ok(Value::Closure(Closure::new(
                    qualified_name,
                    bound_function.map(crate::user_functions::bound_function_binding_for),
                    runmat_types::FunctionSourceRole::Method,
                    bound_function.and_then(crate::user_functions::source_file_for),
                    vec![],
                    None,
                )))
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
