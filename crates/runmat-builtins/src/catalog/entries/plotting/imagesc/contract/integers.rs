use crate::*;

pub(super) const IMAGESC_INTEGER_CDATA: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "CData",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scaled indexed and truecolor CData accept every built-in integer class and remain authoritative on the image object.",
    }];

pub const IMAGESC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = imagesc(integer_CData, ...)",
        inputs: &IMAGESC_INTEGER_CDATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact CData and scaled-mapping provenance are retained while colormap normalization and rendering cross an explicit client boundary.",
    }];
