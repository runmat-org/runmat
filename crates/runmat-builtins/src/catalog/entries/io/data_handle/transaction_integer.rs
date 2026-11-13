//! Transaction integer capabilities mirror queued native values and checked metadata.

use crate::*;

pub(super) const LIFECYCLE_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Transaction id, status, commit, abort, and delete-array consume a managed receiver and text controls; they do not consume or return an integer-class payload.",
};

const VALUES: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "values",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Queued numeric values retain their native integer class until commit casts to the declared array dtype.",
};
const SLICE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "sliceSpec",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "One-based indices and inclusive endpoints accept all integer classes or exactly integral floating values.",
};
const SHAPE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "newShape",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes:
        "Nonnegative dimensions are checked against platform and total-element bounds when queued.",
};
const VALUE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "value",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Scalar integer values retain exact bits in the queue; attribute JSON preserves their mathematical value.",
};
const ATTRS: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "attribute_values",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Scalar integer fields are serialized as exact JSON integers at commit.",
};
const META_SHAPE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "meta.shape",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Schema dimensions are validated as exact platform-sized integers when queued.",
};
const META_CHUNK: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "meta.chunk",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Optional chunk dimensions must be positive and rank-matched to the array shape.",
};

macro_rules! capability {
    ($name:ident, $form:literal, $inputs:expr, $overload:expr, $overflow:expr, $notes:literal) => {
        pub(super) const $name: &[BuiltinIntegerCapabilityDescriptor] =
            &[BuiltinIntegerCapabilityDescriptor {
                form: $form,
                inputs: $inputs,
                computation_domain: BuiltinIntegerComputationDomain::Structural,
                output_class: BuiltinIntegerOutputClassRule::Logical,
                overflow: $overflow,
                backend: BuiltinIntegerBackendRule::HostOnly,
                overload: $overload,
                notes: $notes,
            }];
    };
}

capability!(
    WRITE,
    "tf = DataTransaction.write(tx, arrayName, integer_sliceSpec, integer_values)",
    &[VALUES, SLICE],
    BuiltinIntegerOverloadKind::Multiple,
    BuiltinIntegerOverflowRule::FunctionSpecific,
    "Queueing retains native values and controls; commit applies declared-dtype cast, shape validation, and chunk serialization."
);
capability!(
    SET_ATTR,
    "tf = DataTransaction.set_attr(tx, key, integer_value)",
    &[VALUE],
    BuiltinIntegerOverloadKind::ScalarOnly,
    BuiltinIntegerOverflowRule::NotApplicable,
    "The queued scalar is emitted as an exact JSON integer at commit without floating conversion."
);
capability!(
    SET_ATTRS,
    "tf = DataTransaction.set_attrs(tx, struct_with_integer_values)",
    &[ATTRS],
    BuiltinIntegerOverloadKind::ScalarOnly,
    BuiltinIntegerOverflowRule::NotApplicable,
    "Queued scalar fields become exact JSON integers at commit."
);
capability!(
    RESIZE,
    "tf = DataTransaction.resize(tx, arrayName, integer_newShape)",
    &[SHAPE],
    BuiltinIntegerOverloadKind::StructuralParameter,
    BuiltinIntegerOverflowRule::Error,
    "Exact checked shape is retained and zero-filled storage is recreated at commit."
);
capability!(
    FILL,
    "tf = DataTransaction.fill(tx, arrayName, integer_value, integer_sliceSpec)",
    &[VALUE, SLICE],
    BuiltinIntegerOverloadKind::Multiple,
    BuiltinIntegerOverflowRule::FunctionSpecific,
    "The queued scalar and optional slice retain native class until commit casts once to the declared dtype."
);
capability!(
    CREATE_ARRAY,
    "tf = DataTransaction.create_array(tx, arrayName, struct('shape', integer_shape, 'chunk', integer_chunk))",
    &[META_SHAPE, META_CHUNK],
    BuiltinIntegerOverloadKind::StructuralParameter,
    BuiltinIntegerOverflowRule::Error,
    "Exact shape and chunk dimensions are validated when queued and used to create storage at commit."
);
