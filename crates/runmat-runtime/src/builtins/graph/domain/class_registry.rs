use super::*;

#[derive(Clone, Copy)]
pub(super) enum TriangleMode {
    Full,
    Upper,
    Lower,
}

pub(super) fn ensure_graph_classes_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        register_graph_class(GRAPH_CLASS);
        register_graph_class(DIGRAPH_CLASS);
    });
}

pub(super) fn register_graph_class(name: runmat_types::StaticClassIdentity) {
    let mut properties = HashMap::new();
    for property_name in ["Edges", "Nodes", NUM_NODES_PROPERTY] {
        properties.insert(
            property_name.into(),
            crate::class_registry::RuntimeProperty {
                name: property_name.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
    }
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: name.into(),
        parent: None,
        properties,
        methods: HashMap::<runmat_types::MethodName, crate::class_registry::RuntimeMethod>::new(),
    });
}
