use crate::*;

pub(super) const SPHERE_INTEGER_N_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target calls n a positive integer but does not enumerate typed storage classes. RunMat accepts every exact nonnegative integer class behind a compatibility gate; ordinary host double integer values remain documented behavior.",
    }];

pub const SPHERE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[X,Y,Z] = sphere(integer_n)",
        inputs: &SPHERE_INTEGER_N_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The typed face count is a gated RunMat extension. It is decoded exactly, checked against allocation bounds, and controls only output shape; coordinate generation and statement-form plotting use host double values.",
    }];
