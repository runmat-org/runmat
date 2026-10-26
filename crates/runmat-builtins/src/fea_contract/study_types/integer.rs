//! Integer admission facts for typed study-contract constructors. Counts and
//! byte estimates stay exact; physical coordinates and tolerances cross once
//! into canonical binary64 storage. Values are immutable host objects.

use crate::{
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    ALL_INTEGER_CLASSES,
};

/// Observation constructors that accept only identifiers and typed FEA values.
pub const FEA_OBSERVATION_STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "This constructor accepts identifiers and typed FEA values only.",
    };

/// Sequence constructors that accept only identifiers and typed FEA objects.
pub const FEA_SEQUENCE_STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "This typed sequence constructor accepts identifiers and typed FEA objects only.",
    };

const OBSERVATION_NUMERIC_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sampling, storage, or cost value",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Counts and byte estimates remain exact; physical coordinates and tolerances cross once into binary64.",
    }];

pub const FEA_OBSERVATION_NUMERIC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "typed observation policy",
        inputs: &OBSERVATION_NUMERIC_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Observation authoring constructs immutable host values.",
    }];

const DOMAIN_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "count or coordinates",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes:
        "Mode counts remain exact; physical coordinates cross once into canonical binary64 storage.",
}];

pub const FEA_INDEPENDENT_DOMAIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "independent-domain construction",
        inputs: &DOMAIN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed domains are immutable host authoring values.",
    }];
