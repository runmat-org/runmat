use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerClass {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
}

pub const ALL_INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

pub const SIGNED_INTEGER_CLASSES: [BuiltinIntegerClass; 4] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
];

pub const INTEGER_CLASSES_THROUGH_16_BITS: [BuiltinIntegerClass; 4] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
];

pub const INTEGER_CLASSES_THROUGH_32_BITS: [BuiltinIntegerClass; 6] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
];

pub const UNSIGNED_8_16_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerScalarDoubleRule {
    NotApplicable,
    Allowed,
    AllowedExceptWith64BitInteger,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerInputAvailability {
    Documented,
    RunMatOnly,
    Rejected,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinIntegerInputCapability {
    pub name: &'static str,
    pub classes: &'static [BuiltinIntegerClass],
    pub availability: BuiltinIntegerInputAvailability,
    pub scalar_double: BuiltinIntegerScalarDoubleRule,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerComputationDomain {
    ExactInteger,
    FloatingPoint,
    Predicate,
    Structural,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerOutputClassRule {
    PreserveInput,
    PreserveNondoubleInput,
    Double,
    Logical,
    OptionDependent,
    NotApplicable,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerOverflowRule {
    Saturate,
    Error,
    NotApplicable,
    EvidenceOpen,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerBackendRule {
    HostOnly,
    HostAndGpu,
    GatherFallback,
    GpuRestricted,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerOverloadKind {
    ScalarOnly,
    ElementwiseShapePreserving,
    SameSizeOrScalar,
    BroadcastCompatible,
    StructuralParameter,
    Multiple,
    FunctionSpecific,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinIntegerCapabilityDescriptor {
    pub form: &'static str,
    pub inputs: &'static [BuiltinIntegerInputCapability],
    pub computation_domain: BuiltinIntegerComputationDomain,
    pub output_class: BuiltinIntegerOutputClassRule,
    pub overflow: BuiltinIntegerOverflowRule,
    pub backend: BuiltinIntegerBackendRule,
    pub overload: BuiltinIntegerOverloadKind,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerAuditKind {
    AliasOf,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuiltinIntegerAuditDescriptor {
    pub kind: BuiltinIntegerAuditKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_builtin: Option<&'static str>,
    pub notes: &'static str,
}
