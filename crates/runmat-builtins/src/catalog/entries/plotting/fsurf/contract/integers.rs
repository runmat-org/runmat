use crate::*;

pub(super) const fn integer_input(
    name: &'static str,
    notes: &'static str,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes,
    }
}

pub(super) const INTEGER_DOMAIN_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "xyinterval or uvinterval",
    "RunMat admits exact typed-integer domain bounds only when every value is exactly representable at the binary64 sampling boundary.",
)];

pub(super) const INTEGER_MESH_DENSITY_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "MeshDensity",
    "RunMat admits all eight integer classes as exact bounded host sampling counts.",
)];

pub(super) const INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "ax",
    "MATLAB axes are graphics objects; typed-integer numeric aliases are a gated RunMat representation extension.",
)];

pub(super) const INTEGER_CALLBACK_OUTPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "callback result",
    "RunMat admits scalar typed-integer callback samples only when exactly representable at the binary64 surface-geometry boundary.",
)];

pub(super) const INTEGER_STYLE_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "surface style property",
    "RunMat accepts typed-integer RGB, alpha, and boolean-style controls only behind one independent surface-property gate and validates exact plotting conversion before sampling.",
)];

pub const FSURF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fsurf(f, integer_xyinterval)",
        inputs: &INTEGER_DOMAIN_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The independently gated bounds cross one checked binary64 plotting-domain boundary; the result is an opaque graphics handle.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fsurf(..., 'MeshDensity', integer_density)",
        inputs: &INTEGER_MESH_DENSITY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The independently gated exact count controls bounded host callback sampling and does not become integer output.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fsurf(integer_ax, ...)",
        inputs: &INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility extension gate runs before axes selection and sampling.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fsurf(f_returning_integer, ...)",
        inputs: &INTEGER_CALLBACK_OUTPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Each independently gated authoritative callback sample crosses one checked binary64 surface-geometry boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fsurf(..., property_name, integer_property_value)",
        inputs: &INTEGER_STYLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The independently gated value is validated from authoritative storage before callback sampling and then crosses the applicable renderer-property boundary.",
    },
];
