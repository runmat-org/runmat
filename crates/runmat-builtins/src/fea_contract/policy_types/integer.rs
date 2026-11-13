//! Integer admission facts for typed FEA policy constructors.
//!
//! Physical values (tolerances, coordinates, ordinates, frequencies) admit
//! every built-in integer class and cross once into canonical binary64
//! storage. Counts and resource ceilings stay exact and reject values outside
//! the platform range instead of wrapping or saturating.

use crate::{
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};

const FLOATING_NOTES: &str =
    "Exact host integers are validated, then converted once to canonical binary64 storage; very wide values can round.";
const EXACT_NOTES: &str =
    "Exact host integers and integer-valued doubles are admitted; negative, fractional, or out-of-range values are rejected.";

pub const fn floating(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: FLOATING_NOTES,
    }
}

pub const fn exact(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: EXACT_NOTES,
    }
}

const fn capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
    computation_domain: BuiltinIntegerComputationDomain,
    overflow: BuiltinIntegerOverflowRule,
    notes: &'static str,
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes,
    }
}

/// Physical inputs stored as canonical binary64 values in an immutable host object.
pub const fn physical(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    capability(
        form,
        inputs,
        BuiltinIntegerComputationDomain::FloatingPoint,
        BuiltinIntegerOverflowRule::NotApplicable,
        "The host constructor validates the value and stores canonical binary64 data; no solver runs.",
    )
}

/// Constructors that take no numeric input at all.
pub const NO_NUMERIC_INPUT_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes:
        "This constructor accepts identifiers and typed FEA objects only; it has no numeric input.",
};

/// Closed vocabulary constructors with no arguments.
pub const NO_ARGUMENT_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This constructor takes no arguments and returns one fixed typed value.",
};

pub const TOLERANCE_VALUE_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("value")];
pub const TOLERANCE_VALUE_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [physical("typed tolerance", &TOLERANCE_VALUE_INPUTS)];

pub const ABSOLUTE_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [floating("canonicalValue")];
pub const ABSOLUTE_TOLERANCE_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed absolute tolerance",
    &ABSOLUTE_TOLERANCE_INPUTS,
)];

pub const RESOURCE_CEILING_INPUTS: [BuiltinIntegerInputCapability; 5] = [
    exact("maximumRefinementCycles"),
    exact("maximumSolverTightenings"),
    exact("maximumSamples"),
    exact("maximumWallTimeMilliseconds"),
    exact("maximumResidentBytes"),
];
pub const RESOURCE_CEILING_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [capability(
    "typed accuracy resource ceiling",
    &RESOURCE_CEILING_INPUTS,
    BuiltinIntegerComputationDomain::Structural,
    BuiltinIntegerOverflowRule::Error,
    "Ceilings are stored as exact unsigned integers; a value outside the platform range is an error, never a saturated limit.",
)];

pub const UNIFORM_SAMPLING_INPUTS: [BuiltinIntegerInputCapability; 3] =
    [floating("start"), floating("end"), exact("sampleCount")];
pub const UNIFORM_SAMPLING_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed uniform signal sampling",
    &UNIFORM_SAMPLING_INPUTS,
)];

pub const ADAPTIVE_SAMPLING_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    floating("start"),
    floating("end"),
    floating("maxAbsoluteError"),
    exact("maxSamples"),
];
pub const ADAPTIVE_SAMPLING_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed adaptive signal sampling",
    &ADAPTIVE_SAMPLING_INPUTS,
)];

pub const POINT_SAMPLING_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("points")];
pub const POINT_SAMPLING_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed explicit signal sampling",
    &POINT_SAMPLING_INPUTS,
)];

pub const CONTINUOUS_KNOT_INPUTS: [BuiltinIntegerInputCapability; 2] =
    [floating("abscissa"), floating("value")];
pub const CONTINUOUS_KNOT_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed continuous signal knot",
    &CONTINUOUS_KNOT_INPUTS,
)];

pub const JUMP_KNOT_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    floating("abscissa"),
    floating("leftLimit"),
    floating("rightLimit"),
];
pub const JUMP_KNOT_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [physical("typed jump signal knot", &JUMP_KNOT_INPUTS)];

pub const SPECTRUM_AXIS_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("values")];
pub const SPECTRUM_AXIS_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed response-spectrum axis",
    &SPECTRUM_AXIS_INPUTS,
)];

pub const SPECTRUM_SCALAR_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("value")];
pub const SPECTRUM_SCALAR_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed response-spectrum scalar parameter",
    &SPECTRUM_SCALAR_INPUTS,
)];

pub const SPECTRUM_DIRECTION_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("vector")];
pub const SPECTRUM_DIRECTION_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed response-spectrum direction",
    &SPECTRUM_DIRECTION_INPUTS,
)];

pub const PERCENTAGE_INPUTS: [BuiltinIntegerInputCapability; 2] =
    [floating("primary"), floating("secondary")];
pub const PERCENTAGE_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed directional percentage rule",
    &PERCENTAGE_INPUTS,
)];

pub const TRUNCATION_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("minimumMassRatio")];
pub const TRUNCATION_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed modal truncation policy",
    &TRUNCATION_INPUTS,
)];

pub const GUPTA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    floating("periodicFrequencyHertz"),
    floating("rigidFrequencyHertz"),
];
pub const GUPTA_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [physical("typed Gupta rigid-response policy", &GUPTA_INPUTS)];

pub const SPECTRUM_EVIDENCE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    floating("samplingAdequacyRatio"),
    floating("maximumInterpolationError"),
];
pub const SPECTRUM_EVIDENCE_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed response-spectrum evidence",
    &SPECTRUM_EVIDENCE_INPUTS,
)];

pub const RESPONSE_SPECTRUM_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("ordinates")];
pub const RESPONSE_SPECTRUM_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed response spectrum",
    &RESPONSE_SPECTRUM_INPUTS,
)];
