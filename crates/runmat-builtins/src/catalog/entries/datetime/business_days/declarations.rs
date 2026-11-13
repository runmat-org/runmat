use super::{contracts, documentation, inference};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

pub const HOLIDAYS_LOCAL_YEAR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "holidays-local-year",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "The zero-argument local-year and single-year convenience forms are RunMat extensions until the complete historical holiday list is available",
        error_identifier: Some("RunMat:compatibility:HolidaysLocalYearExtension"),
    };
pub const BUSINESS_DAY_TYPED_LEGACY_SERIAL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "business-day-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Single-precision and typed-integer legacy serial dates in business-day date positions are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:BusinessDayTypedLegacySerialExtension"),
    };

pub const BUSINESS_DAY_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-resident-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Gathering resident numeric calendar input to the host is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatetimeGpuInputExtension"),
    };

pub const BUSINESS_DAY_LOGICAL_NUMERIC_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-logical-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Logical values in numeric date or calendar-control positions are RunMat extensions; logical Weekend masks remain supported",
        error_identifier: Some("RunMat:compatibility:DatetimeLogicalInputExtension"),
    };

pub const BUSINESS_DAY_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    BUSINESS_DAY_TYPED_LEGACY_SERIAL_EXTENSION,
    BUSINESS_DAY_RESIDENT_INPUT_EXTENSION,
    BUSINESS_DAY_LOGICAL_NUMERIC_INPUT_EXTENSION,
];
pub const HOLIDAYS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    HOLIDAYS_LOCAL_YEAR_EXTENSION,
    BUSINESS_DAY_TYPED_LEGACY_SERIAL_EXTENSION,
    BUSINESS_DAY_RESIDENT_INPUT_EXTENSION,
    BUSINESS_DAY_LOGICAL_NUMERIC_INPUT_EXTENSION,
];

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &BUSDAYS_CATALOG_ENTRY,
    &DAYS252BUS_CATALOG_ENTRY,
    &DAYSDIF_CATALOG_ENTRY,
    &FBUSDATE_CATALOG_ENTRY,
    &HOLIDAYS_CATALOG_ENTRY,
    &ISBUSDAY_CATALOG_ENTRY,
    &LBUSDATE_CATALOG_ENTRY,
];

const BASE_EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];
const CLOCK_EFFECTS: &[EffectKind] = &[
    EffectKind::Clock,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    reads_clock: bool,
    extensions: &'static [BuiltinExtensionDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category: "datetime/business-days",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: if reads_clock {
                BuiltinPurity::Impure
            } else {
                BuiltinPurity::Pure
            },
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: if reads_clock {
                CLOCK_EFFECTS
            } else {
                BASE_EFFECTS
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
        integer_capabilities: &[],
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const BUSDAYS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "busdays",
    &contracts::BUSDAYS_DESCRIPTOR,
    documentation::BUSDAYS,
    inference::busdays,
    true,
    &BUSINESS_DAY_EXTENSIONS,
);
pub const DAYS252BUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "days252bus",
    &contracts::DAYS252BUS_DESCRIPTOR,
    documentation::DAYS252BUS,
    inference::days252bus,
    true,
    &BUSINESS_DAY_EXTENSIONS,
);
pub const DAYSDIF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "daysdif",
    &contracts::DAYSDIF_DESCRIPTOR,
    documentation::DAYSDIF,
    inference::daysdif,
    true,
    &BUSINESS_DAY_EXTENSIONS,
);
pub const FBUSDATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fbusdate",
    &contracts::FBUSDATE_DESCRIPTOR,
    documentation::FBUSDATE,
    inference::fbusdate,
    true,
    &BUSINESS_DAY_EXTENSIONS,
);
pub const HOLIDAYS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "holidays",
    &contracts::HOLIDAYS_DESCRIPTOR,
    documentation::HOLIDAYS,
    inference::holidays,
    true,
    &HOLIDAYS_EXTENSIONS,
);
pub const ISBUSDAY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "isbusday",
    &contracts::ISBUSDAY_DESCRIPTOR,
    documentation::ISBUSDAY,
    inference::isbusday,
    true,
    &BUSINESS_DAY_EXTENSIONS,
);
pub const LBUSDATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "lbusdate",
    &contracts::LBUSDATE_DESCRIPTOR,
    documentation::LBUSDATE,
    inference::lbusdate,
    true,
    &BUSINESS_DAY_EXTENSIONS,
);
