use super::*;

pub(super) const DAY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "legacy serial date number",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Legacy numeric input is interpreted at the serial-date boundary and day returns double numeric results or cell-character names.",
}];
pub const DAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "day(integer_serial, dayType)", inputs: &DAY_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "All integer classes enter through authoritative storage; numeric outputs are MATLAB-compatible double arrays. Name modes return cell arrays of character vectors." }];

pub(super) const HOUR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "legacy serial date number",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The merged legacy API documents serial date numbers but does not enumerate typed integer storage. RunMat accepts all eight classes only behind the typed-legacy extension and validates exact storage before conversion.",
}];
pub const HOUR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "hour(integer_serial, F?)", inputs: &HOUR_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer serials are a clearly gated RunMat extension because the documented primary input is datetime and legacy examples use binary64 serial dates. Exact native storage is range-checked before one serial-date conversion; results are host double values with the input shape." }];

pub(super) const YEAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "legacy serial date number",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The retained public legacy API documents serial date numbers but does not enumerate typed integer storage.",
}];
pub const YEAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "year(integer_serial, F?)", inputs: &YEAR_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer serials are gated before one checked serial-date conversion and return host double values with the input shape." }];

pub(super) const MINUTE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "legacy serial date number",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented primary input is datetime and legacy examples use binary64 serial dates. RunMat gates all eight typed integer classes and validates authoritative storage before conversion.",
    }];
pub const MINUTE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "minute(integer_serial, F?)", inputs: &MINUTE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer legacy serials are RunMat-only, cross one checked serial-date boundary, and return host double values with the input shape." }];

pub const SECOND_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "second accepts a datetime value and returns its numeric second component. Native integer host or resident values are not datetime objects and reject without conversion or provider access; legacy serial-date parsing belongs to the separately documented date conversion APIs.",
};

pub(super) const MONTH_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "legacy serial date number",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented primary input is datetime and legacy examples use binary64 serial dates. RunMat gates all eight typed integer classes and validates authoritative storage before conversion.",
    }];
pub const MONTH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "month(integer_serial, F?)", inputs: &MONTH_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer legacy serials are RunMat-only, cross one checked serial-date boundary, and return host double month numbers with the input shape." }];

pub(super) const DATESHIFT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer-valued weekday or occurrence rule",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public API documents integer-valued numeric controls without a per-storage-class table. RunMat's settled compatibility coverage accepts all eight integer classes, reads them exactly before calendar shifting, and also accepts ordinary integer-valued doubles.",
    }];
pub const DATESHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "dateshift(t, ..., integer_weekday_or_rule)", inputs: &DATESHIFT_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The documented contract is the integer-valued numeric control form, not an explicit per-class matrix. RunMat's settled all-eight-class coverage is exact, scalar-expands against datetime arrays, and produces host datetime objects." }];

pub const DATETIME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_YEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_YEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_MONTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_MONTH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_DAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_DAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_HOUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_HOUR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_MINUTE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_MINUTE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_SECOND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SECOND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_BINARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DATETIME_ERRORS,
};
pub const DATESHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATESHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
