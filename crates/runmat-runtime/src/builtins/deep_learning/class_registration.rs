use super::contracts::DLARRAY_CLASS_REGISTERED;
use super::*;

pub(in crate::builtins) fn ensure_dlarray_class_registered() {
    DLARRAY_CLASS_REGISTERED.ensure(|| {
        let methods = [
            runmat_types::StaticMethodName::new("plus"),
            runmat_types::StaticMethodName::new("minus"),
            runmat_types::StaticMethodName::new("times"),
            runmat_types::StaticMethodName::new("rdivide"),
            runmat_types::StaticMethodName::new("mtimes"),
            runmat_types::StaticMethodName::new("sum"),
        ]
        .into_iter()
        .map(|name| {
            (
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("dlarray.{name}"),
                    implicit_class_argument: None,
                },
            )
        })
        .collect::<HashMap<_, _>>();
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: "dlarray".into(),
            parent: None,
            properties: HashMap::new(),
            methods,
        });
    });
}
