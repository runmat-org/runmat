//! Receiver-family integer applicability; payload controls have separate form capabilities.

use crate::*;

pub(super) const DATASET_NONATTRIBUTE_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Dataset identity, array-name lookup/listing, membership, begin, snapshot, and refresh accept a managed Dataset receiver plus text controls; these forms do not consume integer values or indices or return an integer-class payload.",
    };

const STORED_ATTRIBUTE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "stored_attribute",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Manifest JSON retains every integer class's exact mathematical value; JSON does not encode its original narrow storage class.",
};
const DEFAULT_ATTRIBUTE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "defaultValue",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "A missing key returns the supplied default directly, retaining its exact integer class and value.",
};
const VALUE_ATTRIBUTE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "value",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Scalar integer attributes, including wide int64 and uint64, serialize as exact JSON integers without an f64 intermediary.",
};
const VALUES_ATTRIBUTE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "attribute_values",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Scalar integer fields in the attribute struct serialize as exact JSON integers without an f64 intermediary.",
};

pub(super) const DATASET_ATTRS: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "S = Dataset.attrs(ds_with_integer_attributes)",
        inputs: &[STORED_ATTRIBUTE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Stored JSON integers decode exactly as int64 when representable there, otherwise uint64; original narrow classes are not encoded by JSON.",
    }];
pub(super) const DATASET_GET_ATTR: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor {
        form: "value = Dataset.get_attr(ds_with_integer_attribute, key)",
        inputs: &[STORED_ATTRIBUTE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Stored JSON integers decode exactly as int64 where representable, otherwise uint64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "value = Dataset.get_attr(ds, missing_key, integer_defaultValue)",
        inputs: &[DEFAULT_ATTRIBUTE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "A missing key returns the original default Value directly without JSON conversion or floating-point rounding.",
    },
];
pub(super) const DATASET_SET_ATTR: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "tf = Dataset.set_attr(ds, key, integer_value)",
        inputs: &[VALUE_ATTRIBUTE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The scalar is emitted as an exact manifest JSON number; the call returns logical success.",
    }];
pub(super) const DATASET_SET_ATTRS: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "tf = Dataset.set_attrs(ds, struct_with_integer_values)",
        inputs: &[VALUES_ATTRIBUTE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Each scalar integer field is emitted as an exact manifest JSON number; the call returns logical success.",
    }];

pub(super) const ARRAY_METADATA_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "DataArray name, dtype, shape, rank, chunk-shape, and codec accessors take only a managed DataArray receiver. Describing integer storage or returning double-valued shape/rank metadata does not consume, transform, or return an integer-class payload.",
    };

const STORED_ARRAY: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "stored_array",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Declared integer arrays decode directly into same-class real or paired-complex tensor storage.",
};
const SLICE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "sliceSpec",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Scalar one-based indices and inclusive range endpoints accept all eight integer classes or exactly integral floating values.",
};
const ARRAY_VALUES: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "values",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Real or paired-complex integer scalar/tensor values store exactly when their class matches the declared dtype.",
};
const ARRAY_SHAPE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "newShape",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Shape dimensions accept all integer classes or exactly integral floating values within nonnegative platform and checked total-element bounds.",
};
const FILL_VALUE: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "value",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Scalar fill values accept every integer class and retain exact bits when matching the declared array dtype.",
};

pub(super) const ARRAY_READ: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "X = DataArray.read(integer_array, integer_sliceSpec)",
        inputs: &[STORED_ARRAY, SLICE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Full and sliced reads preserve declared integer class, real or paired-complex storage, exact values, shape, and column-major order.",
    }];
pub(super) const ARRAY_WRITE: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "tf = DataArray.write(arr, integer_values) or DataArray.write(arr, integer_sliceSpec, integer_values)",
        inputs: &[ARRAY_VALUES, SLICE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Same-class values retain exact bits; unlike numeric classes are cast once to the declared dtype before chunk serialization, while shape/complexity mismatches reject.",
    }];
pub(super) const ARRAY_RESIZE: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "tf = DataArray.resize(arr, integer_newShape)",
        inputs: &[ARRAY_SHAPE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact shape is checked and persisted before new zero-filled storage is written in the declared dtype.",
    }];
pub(super) const ARRAY_FILL: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "tf = DataArray.fill(arr, integer_value)",
        inputs: &[FILL_VALUE],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "A scalar is cast once to the declared dtype and repeated; nonscalar fill inputs reject.",
    }];
