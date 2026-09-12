use crate::builtins::fea::contracts::descriptors::{
    BOUNDARY_CONDITION_SIGNATURES, COMPARE_SIGNATURES, ERRORS, FIELD_SIGNATURES, PLOT_SIGNATURES,
    RESULTS_SIGNATURES, TRENDS_SIGNATURES,
};
use runmat_builtins::BuiltinCompletionPolicy;
use runmat_builtins::BuiltinDescriptor;
use runmat_builtins::BuiltinIntegerAuditDescriptor;
use runmat_builtins::BuiltinIntegerAuditKind;
use runmat_builtins::BuiltinIntegerBackendRule;
use runmat_builtins::BuiltinIntegerCapabilityDescriptor;
use runmat_builtins::BuiltinIntegerComputationDomain;
use runmat_builtins::BuiltinIntegerInputAvailability;
use runmat_builtins::BuiltinIntegerInputCapability;
use runmat_builtins::BuiltinIntegerOutputClassRule;
use runmat_builtins::BuiltinIntegerOverflowRule;
use runmat_builtins::BuiltinIntegerOverloadKind;
use runmat_builtins::BuiltinIntegerScalarDoubleRule;
use runmat_builtins::BuiltinOutputMode;

const fn fea_floating_input(
    name: &'static str,
    scalar_double: BuiltinIntegerScalarDoubleRule,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double,
        notes: "Host integers cross once into a finite binary64 physics field; provider-resident values are rejected.",
    }
}

const fn fea_floating_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
    overload: BuiltinIntegerOverloadKind,
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload,
        notes: "The RunMat-native constructor validates its host value and performs one explicit IEEE-754 binary64 model-storage conversion; wide integers can round.",
    }
}

pub(in crate::builtins::fea) const MATERIAL_MECHANICAL_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "mechanical numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub(in crate::builtins::fea) const MATERIAL_THERMAL_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "thermal numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub(in crate::builtins::fea) const MATERIAL_ACOUSTIC_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "acoustic numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub(in crate::builtins::fea) const MATERIAL_ELECTRICAL_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "electrical numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub(in crate::builtins::fea) const MATERIAL_RESPONSE_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "conductivity response numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub(in crate::builtins::fea) const MATERIAL_PLASTIC_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "plastic numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub const FEA_MATERIAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 6] = [
    fea_floating_capability(
        "mechanical material fields",
        &MATERIAL_MECHANICAL_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "thermal material fields",
        &MATERIAL_THERMAL_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "acoustic material fields",
        &MATERIAL_ACOUSTIC_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "electrical material fields",
        &MATERIAL_ELECTRICAL_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "electrical frequency-response fields",
        &MATERIAL_RESPONSE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "plastic material fields",
        &MATERIAL_PLASTIC_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
];

pub(in crate::builtins::fea) const LOAD_VECTOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "three-element vector",
        BuiltinIntegerScalarDoubleRule::NotApplicable,
    )];
pub(in crate::builtins::fea) const LOAD_SCALAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "scalar physics fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
pub(in crate::builtins::fea) const LOAD_CURRENT_DENSITY_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 2] = [
    fea_floating_input(
        "three-element vector",
        BuiltinIntegerScalarDoubleRule::NotApplicable,
    ),
    fea_floating_input(
        "phase and amplitude",
        BuiltinIntegerScalarDoubleRule::Allowed,
    ),
];
pub const FEA_LOAD_CASE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 7] = [
    fea_floating_capability(
        "force vector",
        &LOAD_VECTOR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::FunctionSpecific,
    ),
    fea_floating_capability(
        "moment or torque vector",
        &LOAD_VECTOR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::FunctionSpecific,
    ),
    fea_floating_capability(
        "pressure magnitude",
        &LOAD_SCALAR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "body-force vector",
        &LOAD_VECTOR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::FunctionSpecific,
    ),
    fea_floating_capability(
        "current-density fields",
        &LOAD_CURRENT_DENSITY_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "coil-current fields",
        &LOAD_SCALAR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "volumetric heat source",
        &LOAD_SCALAR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
];

pub(in crate::builtins::fea) const DOMAIN_FLOATING_INTEGER_INPUTS: [BuiltinIntegerInputCapability;
    1] = [fea_floating_input(
    "domain numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub(in crate::builtins::fea) const DOMAIN_REVISION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "field_source.revision",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "The structural revision is decoded exactly as u32 and rejects negative or out-of-range integer values.",
}];
pub const FEA_DOMAIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    fea_floating_capability("thermo-mechanical physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
    BuiltinIntegerCapabilityDescriptor { form: "thermo field-source revision", inputs: &DOMAIN_REVISION_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The RunMat-native constructor preserves the exact u32 revision in both typed model storage and its public object representation." },
    fea_floating_capability("electro-thermal physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
    fea_floating_capability("electromagnetic physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
    fea_floating_capability("CFD physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
];

pub(in crate::builtins::fea) const INTERFACE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "interface numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
pub const FEA_INTERFACE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    fea_floating_capability(
        "contact interface fields",
        &INTERFACE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "fluid-structure interface fields",
        &INTERFACE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "conjugate-heat-transfer interface fields",
        &INTERFACE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
];

pub const FEA_STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This RunMat-native FEA API accepts object, text, or enum inputs rather than numeric data; nested typed objects retain their already-defined numeric contracts and no provider gather occurs.",
};

pub(in crate::builtins::fea) const RUN_OPTIONS_EXACT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "structural iteration, step, retry, mode, and refresh counts",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host integer scalars and integral scalar doubles are range-checked and decoded exactly as usize; provider-resident values are rejected.",
    }];
pub(in crate::builtins::fea) const RUN_OPTIONS_FLOATING_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "tolerance, timing, residual, convergence, and frequency fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];

const fn run_options_exact_capability(form: &'static str) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &RUN_OPTIONS_EXACT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact count is preserved in the typed run-options payload and public object representation.",
    }
}

const fn run_options_floating_capability(form: &'static str) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &RUN_OPTIONS_FLOATING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Floating solver controls use finite IEEE-754 binary64 storage; wide integer inputs can round while structural counts remain exact.",
    }
}

pub const FEA_RUN_OPTIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 18] = [
    run_options_exact_capability("modal structural controls"),
    run_options_floating_capability("modal floating controls"),
    run_options_exact_capability("acoustic structural controls"),
    run_options_floating_capability("acoustic floating controls"),
    run_options_exact_capability("thermal structural controls"),
    run_options_floating_capability("thermal floating controls"),
    run_options_exact_capability("transient structural controls"),
    run_options_floating_capability("transient floating controls"),
    run_options_exact_capability("CFD structural controls"),
    run_options_floating_capability("CFD floating controls"),
    run_options_exact_capability("CHT structural controls"),
    run_options_floating_capability("CHT floating controls"),
    run_options_exact_capability("FSI structural controls"),
    run_options_floating_capability("FSI floating controls"),
    run_options_exact_capability("nonlinear structural controls"),
    run_options_floating_capability("nonlinear floating controls"),
    run_options_exact_capability("electromagnetic structural controls"),
    run_options_floating_capability("electromagnetic floating controls"),
];

pub(in crate::builtins::fea) const RESULTS_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "one-based result indices",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host numeric scalars or vectors are decoded exactly, require positive one-based indices, preserve order and duplicates, and reject matrix and provider-resident inputs.",
    }];
pub(in crate::builtins::fea) const RESULTS_FLAG_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric inclusion predicate",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host scalar logical values and exact numeric zero or one are accepted; every other numeric value and provider-resident input is rejected.",
    }];
pub const FEA_RESULTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "ModeIndices", inputs: &RESULTS_SELECTOR_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The one-based public selector is translated once to the operation layer's zero-based index and public structural result fields remain exact." },
    BuiltinIntegerCapabilityDescriptor { form: "TransientSnapshotIndices", inputs: &RESULTS_SELECTOR_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The one-based public selector is translated once to the operation layer's zero-based index and public structural result fields remain exact." },
    BuiltinIntegerCapabilityDescriptor { form: "numeric inclusion predicates", inputs: &RESULTS_FLAG_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Logical and numeric zero/one select query projections without converting provider data." },
];

pub(in crate::builtins::fea) const TRENDS_WINDOW_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "WindowSize",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive host integer scalar or integral scalar double is decoded exactly as usize; provider-resident values are rejected.",
    }];
pub const FEA_TRENDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "WindowSize", inputs: &TRENDS_WINDOW_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The positive window size and structural trend counts remain exact in the public result object; time and rate fields remain binary64." }];
pub const FEA_BOUNDARY_CONDITION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOUNDARY_CONDITION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const fn boundary_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "An exact scalar is converted once to the model's binary64 storage field using Rust's IEEE-754 integer-to-f64 conversion.",
    }
}

pub(in crate::builtins::fea) const BOUNDARY_ROTATION_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 3] = [
    boundary_integer_input("rx"),
    boundary_integer_input("ry"),
    boundary_integer_input("rz"),
];
pub(in crate::builtins::fea) const BOUNDARY_IMPEDANCE_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [boundary_integer_input("specificImpedancePaSPerM")];
pub(in crate::builtins::fea) const BOUNDARY_TEMPERATURE_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [boundary_integer_input("temperatureK")];
pub(in crate::builtins::fea) const BOUNDARY_HEAT_FLUX_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 1] = [boundary_integer_input("heatFluxWPerM2")];
pub(in crate::builtins::fea) const BOUNDARY_CONVECTION_INTEGER_INPUTS:
    [BuiltinIntegerInputCapability; 2] = [
    boundary_integer_input("ambientTemperatureK"),
    boundary_integer_input("coefficientWPerM2K"),
];
pub(in crate::builtins::fea) const BOUNDARY_INLET_INTEGER_INPUTS: [BuiltinIntegerInputCapability;
    1] = [boundary_integer_input("velocityMPerS")];
pub(in crate::builtins::fea) const BOUNDARY_OUTLET_INTEGER_INPUTS: [BuiltinIntegerInputCapability;
    1] = [boundary_integer_input("pressurePa")];

const fn boundary_integer_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The constructor validates scalar shape and finiteness, then performs one explicit IEEE-754 binary64 model-storage conversion; wide integers can therefore round.",
    }
}

pub const FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 7] = [
    boundary_integer_capability(
        "prescribedRotation integer fields",
        &BOUNDARY_ROTATION_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "acousticImpedance integer field",
        &BOUNDARY_IMPEDANCE_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "thermalPrescribedTemperature integer field",
        &BOUNDARY_TEMPERATURE_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "thermalHeatFlux integer field",
        &BOUNDARY_HEAT_FLUX_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "thermalConvection integer fields",
        &BOUNDARY_CONVECTION_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "cfdInletVelocity integer field",
        &BOUNDARY_INLET_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "cfdOutletPressure integer field",
        &BOUNDARY_OUTLET_INTEGER_INPUTS,
    ),
];
pub const FEA_RESULTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESULTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_FIELD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIELD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_PLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_COMPARE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPARE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_TRENDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRENDS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
