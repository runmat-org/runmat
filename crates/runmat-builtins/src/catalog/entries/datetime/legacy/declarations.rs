use super::{contracts, documentation, inference, integer};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &DATENUM_CATALOG_ENTRY,
    &DATESTR_CATALOG_ENTRY,
    &DATEVEC_CATALOG_ENTRY,
    &EOMDAY_CATALOG_ENTRY,
    &ETIME_CATALOG_ENTRY,
    &ISBETWEEN_CATALOG_ENTRY,
    &WEEKDAY_CATALOG_ENTRY,
];

pub const LEGACY_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-resident-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Gathering resident numeric date input to host is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatetimeGpuInputExtension"),
    };
pub const LEGACY_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-logical-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Logical values in numeric date positions are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DatetimeLogicalInputExtension"),
};
pub const LEGACY_TYPED_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "legacy-datetime-typed-numeric-input", mode: BuiltinExtensionMode::RunMatOnly,
    description: "Single and integer numeric legacy date inputs beyond documented double forms are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LegacyDatetimeTypedNumericExtension"),
};
pub const DATENUM_PARTIAL_COMPONENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datenum-four-five-components",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Four- and five-component datenum calls are retained RunMat convenience forms",
        error_identifier: Some("RunMat:compatibility:DatenumPartialComponentsExtension"),
    };
pub const ISBETWEEN_SERIAL_DATE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isbetween-serial-date-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "Numeric serial-date and text operands for isbetween are RunMat convenience forms",
        error_identifier: Some("RunMat:compatibility:IsbetweenSerialDateExtension"),
    };
const BASE_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    LEGACY_RESIDENT_INPUT_EXTENSION,
    LEGACY_LOGICAL_INPUT_EXTENSION,
];
const TYPED_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    LEGACY_RESIDENT_INPUT_EXTENSION,
    LEGACY_LOGICAL_INPUT_EXTENSION,
    LEGACY_TYPED_NUMERIC_EXTENSION,
];
const DATENUM_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    LEGACY_RESIDENT_INPUT_EXTENSION,
    LEGACY_LOGICAL_INPUT_EXTENSION,
    LEGACY_TYPED_NUMERIC_EXTENSION,
    DATENUM_PARTIAL_COMPONENT_EXTENSION,
];
const BETWEEN_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    LEGACY_RESIDENT_INPUT_EXTENSION,
    LEGACY_LOGICAL_INPUT_EXTENSION,
    ISBETWEEN_SERIAL_DATE_EXTENSION,
];

const CLOCK_EFFECTS: &[EffectKind] = &[
    EffectKind::Clock,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const HOST_EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];

const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    may_read_clock: bool,
    extensions: &'static [BuiltinExtensionDescriptor],
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "datetime/legacy",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: if may_read_clock {
                BuiltinPurity::Impure
            } else {
                BuiltinPurity::Pure
            },
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: if may_read_clock {
                CLOCK_EFFECTS
            } else {
                HOST_EFFECTS
            },
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::GatherToHost,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions,
        integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const DATENUM_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "datenum",
    &contracts::DATENUM,
    documentation::DATENUM,
    inference::datenum,
    true,
    DATENUM_EXTENSIONS,
    &integer::DATENUM,
);
pub const DATESTR_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "datestr",
    &contracts::DATESTR,
    documentation::DATESTR,
    inference::datestr,
    true,
    TYPED_EXTENSIONS,
    &integer::DATESTR,
);
pub const DATEVEC_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "datevec",
    &contracts::DATEVEC,
    documentation::DATEVEC,
    inference::datevec,
    true,
    TYPED_EXTENSIONS,
    &integer::DATEVEC,
);
pub const EOMDAY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "eomday",
    &contracts::EOMDAY,
    documentation::EOMDAY,
    inference::eomday,
    false,
    BASE_EXTENSIONS,
    &integer::EOMDAY,
);
pub const ETIME_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "etime",
    &contracts::ETIME,
    documentation::ETIME,
    inference::etime,
    false,
    TYPED_EXTENSIONS,
    &integer::ETIME,
);
pub const ISBETWEEN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "isbetween",
    &contracts::ISBETWEEN,
    documentation::ISBETWEEN,
    inference::isbetween,
    true,
    BETWEEN_EXTENSIONS,
    &integer::ISBETWEEN,
);
pub const WEEKDAY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "weekday",
    &contracts::WEEKDAY,
    documentation::WEEKDAY,
    inference::weekday,
    true,
    BASE_EXTENSIONS,
    &integer::WEEKDAY,
);
