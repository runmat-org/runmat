use super::*;

pub(crate) fn register_delaunaytri_class() {
    ensure_registered();
}

pub(super) fn ensure_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        let mut properties = HashMap::new();
        for name in [
            LEGACY_POINTS_PROPERTY,
            LEGACY_CONNECTIVITY_PROPERTY,
            CONSTRAINTS_PROPERTY,
            POINTS_PROPERTY,
            CONNECTIVITY_PROPERTY,
            DIMENSION_PROPERTY,
        ] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: None,
                },
            );
        }

        let mut methods = HashMap::new();
        for name in [
            runmat_types::StaticMethodName::new("freeBoundary"),
            runmat_types::StaticMethodName::new("nearestNeighbor"),
            runmat_types::StaticMethodName::new("pointLocation"),
        ] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{DELAUNAY_TRI_CLASS}.{name}"),
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
        }

        crate::class_registry::register_builtin_class(
            DELAUNAY_TRI_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: DELAUNAY_TRI_CLASS.into(),
                parent: None,
                properties,
                methods,
            },
        );
    });
}
