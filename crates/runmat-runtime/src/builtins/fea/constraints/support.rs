pub(crate) mod direction;
pub(crate) mod dof;
pub(crate) mod frame;
pub(crate) mod pairing;
pub(crate) mod periodic_transform;
pub(crate) mod reference_point;

use std::collections::HashMap;

use runmat_types::MemberAccess;

pub(super) use direction::direction_from_value;
pub(super) use dof::structural_dof_values_from_value;
pub(super) use frame::coordinate_frame_from_value;
pub(super) use pairing::pairing_policy_from_value;
pub(super) use periodic_transform::periodic_transform_from_value;
pub(super) use reference_point::reference_point_from_value;

pub(super) fn ensure_constraint_support_classes_registered() {
    dof::ensure_dof_classes_registered();
    direction::ensure_direction_class_registered();
    frame::ensure_frame_classes_registered();
    pairing::ensure_pairing_classes_registered();
    periodic_transform::ensure_periodic_transform_classes_registered();
    reference_point::ensure_reference_point_class_registered();
}

fn register_runtime_class(
    name: &str,
    fields: &[&str],
    methods: HashMap<String, crate::class_registry::RuntimeMethod>,
) {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: name.into(),
        parent: None,
        properties: fields
            .iter()
            .map(|field| {
                (
                    (*field).into(),
                    crate::class_registry::RuntimeProperty {
                        name: (*field).into(),
                        is_static: false,
                        is_constant: true,
                        is_dependent: false,
                        get_access: MemberAccess::Public,
                        set_access: MemberAccess::Private,
                        default_value: None,
                    },
                )
            })
            .collect(),
        methods,
    });
}

fn static_method(name: &str, function_name: &str) -> crate::class_registry::RuntimeMethod {
    crate::class_registry::RuntimeMethod {
        name: name.into(),
        is_static: true,
        is_abstract: false,
        is_sealed: false,
        access: MemberAccess::Public,
        function_name: function_name.into(),
        implicit_class_argument: None,
    }
}
