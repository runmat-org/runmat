//! Runtime class metadata for the GC-owned Map handle.

use super::*;
use runmat_types::MemberAccess;
use std::collections::HashMap;

static CONTAINERS_MAP_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS_IDENTITY);

pub(super) fn ensure_containers_map_class_registered() {
    CONTAINERS_MAP_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in ["Count", "KeyType", "ValueType"] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
        }

        let mut methods = HashMap::new();
        for (name, function_name) in [
            (runmat_types::StaticMethodName::new("keys"), BUILTIN_KEYS),
            (
                runmat_types::StaticMethodName::new("values"),
                BUILTIN_VALUES,
            ),
            (runmat_types::StaticMethodName::new("isKey"), BUILTIN_IS_KEY),
            (
                runmat_types::StaticMethodName::new("remove"),
                BUILTIN_REMOVE,
            ),
            (crate::OBJECT_SUBSREF_METHOD, BUILTIN_SUBSREF),
            (crate::OBJECT_SUBSASGN_METHOD, BUILTIN_SUBSASGN),
        ] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: function_name.into(),
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
        }

        crate::class_registry::register_builtin_class(
            CLASS_IDENTITY,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: CLASS_IDENTITY.into(),
                parent: None,
                properties,
                methods,
            },
        );
    });
}
