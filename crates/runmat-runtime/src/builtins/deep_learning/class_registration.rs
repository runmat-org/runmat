use super::contracts::DLARRAY_CLASS_REGISTERED;
use super::*;
use runmat_types::MemberAccess;

static DLNETWORK_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(runmat_types::standard::DL_NETWORK);
static SERIES_NETWORK_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(runmat_types::standard::SERIES_NETWORK);
static DAG_NETWORK_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(runmat_types::standard::DAG_NETWORK);

pub(in crate::builtins) fn ensure_dlarray_class_registered() {
    DLARRAY_CLASS_REGISTERED.ensure(|| {
        let methods =
            crate::class_registry::standard_runtime_methods(runmat_types::standard::DL_ARRAY);
        crate::class_registry::register_builtin_class(
            runmat_types::standard::DL_ARRAY,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: runmat_types::standard::DL_ARRAY.owned(),
                parent: None,
                properties: HashMap::new(),
                methods,
            },
        );
    });
}

pub(in crate::builtins) fn ensure_network_class_registered(
    identity: runmat_types::StaticClassIdentity,
) {
    let registration = if identity == runmat_types::standard::DL_NETWORK {
        &DLNETWORK_REGISTERED
    } else if identity == runmat_types::standard::SERIES_NETWORK {
        &SERIES_NETWORK_REGISTERED
    } else if identity == runmat_types::standard::DAG_NETWORK {
        &DAG_NETWORK_REGISTERED
    } else {
        return;
    };
    registration.ensure(|| {
        let mut properties = HashMap::new();
        for name in [
            "Layers",
            "LayerNames",
            "InputNames",
            "OutputNames",
            "Connections",
            "Initialized",
            "Initialize",
            "SupportedExecution",
            "TrainingInfo",
            "TrainingLoss",
            "TrainedWith",
        ] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
        }
        properties.insert(
            super::model::state_storage_property().into(),
            crate::class_registry::RuntimeProperty {
                name: super::model::state_storage_property().into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Private,
                set_access: MemberAccess::Private,
                default_value: None,
            },
        );
        for name in ["Learnables", "State"] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: None,
                },
            );
        }
        let mut methods = HashMap::new();
        for name in ["get.Learnables", "set.Learnables", "get.State", "set.State"] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: name.into(),
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
        }
        crate::class_registry::register_builtin_class(
            identity,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: identity.owned(),
                parent: None,
                properties,
                methods,
            },
        );
    });
}
