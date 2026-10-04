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
