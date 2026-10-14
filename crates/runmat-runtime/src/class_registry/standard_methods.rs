//! Executable projection of immutable standard-class method declarations.
//!
//! Domain owners still provide live property defaults and registration timing.

use std::collections::HashMap;

use runmat_types::{CallableIdentity, MethodName, StaticClassIdentity};

use super::RuntimeMethod;

pub(crate) fn standard_runtime_methods(
    class: StaticClassIdentity,
) -> HashMap<MethodName, RuntimeMethod> {
    runmat_builtins::standard_class_declaration(&class.owned())
        .expect("registered standard class has an immutable declaration")
        .methods
        .into_iter()
        .map(|method| {
            let function_name = match method.callable {
                CallableIdentity::Builtin(id) => id.0,
                CallableIdentity::ExternalName(name) => name
                    .display_name()
                    .expect("standard class method has a qualified callable identity"),
                identity => panic!(
                    "standard class {class} method '{}' has unsupported callable identity {identity:?}",
                    method.name
                ),
            };
            let name = method.name;
            (
                name.clone(),
                RuntimeMethod {
                    name,
                    is_static: method.is_static,
                    is_abstract: method.attributes.is_abstract,
                    is_sealed: method.attributes.is_sealed,
                    access: method.attributes.access,
                    function_name,
                    implicit_class_argument: method.implicit_class_argument,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::standard;

    #[test]
    fn standard_runtime_methods_preserve_canonical_members_and_callables() {
        for class in [
            standard::GPU_ARRAY,
            standard::DURATION,
            standard::DATETIME,
            standard::CALENDAR_DURATION,
            standard::DL_ARRAY,
        ] {
            let declaration = runmat_builtins::standard_class_declaration(&class.owned()).unwrap();
            let methods = standard_runtime_methods(class);
            assert_eq!(methods.len(), declaration.methods.len());
            for declared in declaration.methods {
                let actual = &methods[&declared.name];
                assert_eq!(actual.name, declared.name);
                assert_eq!(actual.is_static, declared.is_static);
                assert_eq!(actual.is_abstract, declared.attributes.is_abstract);
                assert_eq!(actual.is_sealed, declared.attributes.is_sealed);
                assert_eq!(actual.access, declared.attributes.access);
                assert_eq!(
                    actual.implicit_class_argument,
                    declared.implicit_class_argument
                );
                assert_eq!(
                    actual.function_name,
                    declared.callable.display_name().unwrap()
                );
            }
        }
    }
}
