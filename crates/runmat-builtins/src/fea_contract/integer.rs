//! Integer boundary facts for the FEA operation surface. Runtime entrypoints
//! re-export these values.

use crate::BuiltinIntegerAuditDescriptor;
use crate::BuiltinIntegerAuditKind;
use crate::BuiltinIntegerBackendRule;
use crate::BuiltinIntegerCapabilityDescriptor;
use crate::BuiltinIntegerComputationDomain;
use crate::BuiltinIntegerInputAvailability;
use crate::BuiltinIntegerInputCapability;
use crate::BuiltinIntegerOutputClassRule;
use crate::BuiltinIntegerOverflowRule;
use crate::BuiltinIntegerOverloadKind;
use crate::BuiltinIntegerScalarDoubleRule;

pub const fn fea_floating_input(
    name: &'static str,
    scalar_double: BuiltinIntegerScalarDoubleRule,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double,
        notes: "Host integers cross once into a finite binary64 physics field; provider-resident values are rejected.",
    }
}

pub const fn fea_floating_capability(
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

pub const MATERIAL_MECHANICAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "mechanical numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
pub const MATERIAL_THERMAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "thermal numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
pub const MATERIAL_ACOUSTIC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "acoustic numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
pub const MATERIAL_ELECTRICAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "electrical numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
pub const MATERIAL_RESPONSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "conductivity response numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
pub const MATERIAL_PLASTIC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
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

pub const FEA_STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This RunMat-native FEA API accepts object, text, or enum inputs rather than numeric data; nested typed objects retain their already-defined numeric contracts and no provider gather occurs.",
};

pub const RESULTS_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "one-based result indices",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host numeric scalars or vectors are decoded exactly, require positive one-based indices, preserve order and duplicates, and reject matrix and provider-resident inputs.",
    }];
pub const RESULTS_FLAG_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric inclusion predicate",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host scalar logical values and exact numeric zero or one are accepted; every other numeric value and provider-resident input is rejected.",
    }];
pub const FEA_RESULTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "ModeIndices", inputs: &RESULTS_SELECTOR_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The one-based public selector is translated once to the operation layer's zero-based index and public structural result fields remain exact." },
    BuiltinIntegerCapabilityDescriptor { form: "TransientSnapshotIndices", inputs: &RESULTS_SELECTOR_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The one-based public selector is translated once to the operation layer's zero-based index and public structural result fields remain exact." },
    BuiltinIntegerCapabilityDescriptor { form: "numeric inclusion predicates", inputs: &RESULTS_FLAG_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Logical and numeric zero/one select query projections without converting provider data." },
];

pub const TRENDS_WINDOW_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "WindowSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive host integer scalar or integral scalar double is decoded exactly as usize; provider-resident values are rejected.",
    }];
pub const FEA_TRENDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "WindowSize", inputs: &TRENDS_WINDOW_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The positive window size and structural trend counts remain exact in the public result object; time and rate fields remain binary64." }];

pub const FIELD_PAGE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "start",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive one-based host integer or integral scalar double; provider-resident values are rejected.",
    },
    BuiltinIntegerInputCapability {
        name: "count",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive host integer or integral scalar double, capped by the field-page contract; provider-resident values are rejected.",
    },
];
pub const FEA_FIELD_PAGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "field page bounds",
        inputs: &FIELD_PAGE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Paging indexes and counts remain exact native integers; numerical field values retain their real or complex floating representation.",
    }];
