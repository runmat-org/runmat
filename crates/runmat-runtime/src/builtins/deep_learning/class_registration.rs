use super::contracts::DLARRAY_CLASS_REGISTERED;
use super::*;

pub(in crate::builtins) fn ensure_dlarray_class_registered() {
    DLARRAY_CLASS_REGISTERED.ensure(|| {
        let methods =
            crate::class_registry::standard_runtime_methods(runmat_types::standard::DL_ARRAY);
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: runmat_types::standard::DL_ARRAY.owned(),
            parent: None,
            properties: HashMap::new(),
            methods,
        });
    });
}
