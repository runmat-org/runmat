pub(crate) mod direction;
pub(crate) mod dof;
pub(crate) mod frame;
pub(crate) mod motion;
pub(crate) mod pairing;
pub(crate) mod periodic_transform;
pub(crate) mod reference_point;

pub(super) use direction::direction_from_value;
pub(super) use dof::structural_dof_values_from_value;
pub(crate) use frame::coordinate_frame_from_value;
pub(super) use motion::structural_motions_from_value;
pub(super) use pairing::pairing_policy_from_value;
pub(super) use periodic_transform::periodic_transform_from_value;
pub(super) use reference_point::reference_point_from_value;

pub(super) fn ensure_constraint_support_classes_registered() {
    dof::ensure_dof_classes_registered();
    direction::ensure_direction_class_registered();
    frame::ensure_frame_classes_registered();
    motion::ensure_motion_classes_registered();
    pairing::ensure_pairing_classes_registered();
    periodic_transform::ensure_periodic_transform_classes_registered();
    reference_point::ensure_reference_point_class_registered();
}
