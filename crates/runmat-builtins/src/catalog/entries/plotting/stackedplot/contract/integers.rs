use crate::*;

pub(super) const STACKEDPLOT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X, Y, or numeric table variables",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented numeric data surface includes native integer arrays. Exact source class, shape, and values remain authoritative independently of floating render geometry.",
    }];

pub(super) const STACKEDPLOT_INTEGER_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "vars or XVariable",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents integer arrays as table-variable selectors. RunMat decodes them as exact one-based structural indices.",
    }];

pub const STACKEDPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "s = stackedplot(integer_Y) or stackedplot(integer_X, integer_Y)",
        inputs: &STACKEDPLOT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Graphics source properties retain native storage and rendering crosses an explicit floating geometry boundary. Automatic residency gathers transparently; explicit gpuArray input is separately gated because the public page does not document it.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = stackedplot(tbl, integer_vars) or stackedplot(tbl, XVariable=integer_xvar)",
        inputs: &STACKEDPLOT_INTEGER_SELECTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Selectors are validated exactly as positive in-range one-based indices; selected integer variables retain their source classes and values.",
    },
];
