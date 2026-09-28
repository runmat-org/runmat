use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contracts;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub const HOLIDAYS_LOCAL_YEAR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "holidays-local-year",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "The zero-argument local-year and single-year convenience forms are RunMat extensions until the complete historical holiday list is available",
        error_identifier: Some("RunMat:compatibility:HolidaysLocalYearExtension"),
    };
pub const HOLIDAYS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HOLIDAYS_LOCAL_YEAR_EXTENSION];

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
    false,
    &[],
);
pub const DAYS252BUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "days252bus",
    &contracts::DAYS252BUS_DESCRIPTOR,
    documentation::DAYS252BUS,
    inference::days252bus,
    false,
    &[],
);
pub const DAYSDIF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "daysdif",
    &contracts::DAYSDIF_DESCRIPTOR,
    documentation::DAYSDIF,
    inference::daysdif,
    false,
    &[],
);
pub const FBUSDATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fbusdate",
    &contracts::FBUSDATE_DESCRIPTOR,
    documentation::FBUSDATE,
    inference::fbusdate,
    false,
    &[],
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
    false,
    &[],
);
pub const LBUSDATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "lbusdate",
    &contracts::LBUSDATE_DESCRIPTOR,
    documentation::LBUSDATE,
    inference::lbusdate,
    false,
    &[],
);
