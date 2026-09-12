use super::*;

pub(super) fn ensure_geometry_classes_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: GEOMETRY_ASSET_CLASS.into(),
            parent: None,
            properties: HashMap::new(),
            methods: geometry_asset_methods(),
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: GEOMETRY_INSPECT_RESULT_CLASS.into(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::<runmat_types::MethodName, crate::class_registry::RuntimeMethod>::new(
            ),
        });
        triangulation::register_delaunaytri_class();
    });
}

pub(super) fn geometry_asset_methods(
) -> HashMap<runmat_types::MethodName, crate::class_registry::RuntimeMethod> {
    [
        (
            runmat_types::StaticMethodName::new("listRegions"),
            GEOMETRY_LIST_REGIONS_NAME,
        ),
        (
            runmat_types::StaticMethodName::new("meshes"),
            GEOMETRY_MESHES_NAME,
        ),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.into(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}
