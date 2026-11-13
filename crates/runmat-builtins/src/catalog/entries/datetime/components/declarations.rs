use super::{documentation, inference};
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

// Relative text and offset text can consult the owned calendar policy even
// when the ordinary numeric/datetime overload does not read the current time.
macro_rules! entry {
    ($constant:ident, $name:literal, $descriptor:ident, $documentation:ident, $infer:ident,
     $async:ident, $residency:ident, $effects:expr, $extensions:expr, $integers:expr) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "datetime/components",
            documentation: documentation::$documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                    inference::$infer,
                )),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::$async,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: $effects,
                capabilities: &[],
            },
            placement: BuiltinPlacementContract {
                preparation: BuiltinOperandPreparation::RuntimeOwned,
                portability: BuiltinPortability::NativeAndWasm,
                accelerator: BuiltinAcceleratorPolicy::Forbidden,
                residency: BuiltinResidencyPolicy::$residency,
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
            extensions: $extensions,
            integer_capabilities: $integers,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}
const COMPONENT_EFFECTS: &[EffectKind] = &[
    EffectKind::Clock,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const CURRENT_EFFECTS: &[EffectKind] = &[EffectKind::Clock, EffectKind::MayThrow];

entry!(
    YEAR_CATALOG_ENTRY,
    "year",
    DATETIME_YEAR_DESCRIPTOR,
    YEAR,
    year,
    MaySuspend,
    GatherToHost,
    COMPONENT_EFFECTS,
    &YEAR_EXTENSIONS,
    &YEAR_INTEGER_CAPABILITIES
);
entry!(
    MONTH_CATALOG_ENTRY,
    "month",
    DATETIME_MONTH_DESCRIPTOR,
    MONTH,
    month,
    MaySuspend,
    GatherToHost,
    COMPONENT_EFFECTS,
    &MONTH_EXTENSIONS,
    &MONTH_INTEGER_CAPABILITIES
);
entry!(
    DAY_CATALOG_ENTRY,
    "day",
    DATETIME_DAY_DESCRIPTOR,
    DAY,
    day,
    MaySuspend,
    GatherToHost,
    COMPONENT_EFFECTS,
    &DAY_EXTENSIONS,
    &DAY_INTEGER_CAPABILITIES
);
entry!(
    HOUR_CATALOG_ENTRY,
    "hour",
    DATETIME_HOUR_DESCRIPTOR,
    HOUR,
    hour,
    MaySuspend,
    GatherToHost,
    COMPONENT_EFFECTS,
    &HOUR_EXTENSIONS,
    &HOUR_INTEGER_CAPABILITIES
);
entry!(
    MINUTE_CATALOG_ENTRY,
    "minute",
    DATETIME_MINUTE_DESCRIPTOR,
    MINUTE,
    minute,
    MaySuspend,
    GatherToHost,
    COMPONENT_EFFECTS,
    &MINUTE_EXTENSIONS,
    &MINUTE_INTEGER_CAPABILITIES
);
entry!(
    NOW_CATALOG_ENTRY,
    "now",
    NOW_DESCRIPTOR,
    NOW,
    now,
    NeverSuspends,
    Host,
    CURRENT_EFFECTS,
    &[],
    &[]
);
entry!(
    TODAY_CATALOG_ENTRY,
    "today",
    TODAY_DESCRIPTOR,
    TODAY,
    today,
    NeverSuspends,
    Host,
    CURRENT_EFFECTS,
    &[],
    &[]
);
entry!(
    CLOCK_CATALOG_ENTRY,
    "clock",
    CLOCK_DESCRIPTOR,
    CLOCK,
    clock,
    NeverSuspends,
    Host,
    CURRENT_EFFECTS,
    &[],
    &[]
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &YEAR_CATALOG_ENTRY,
    &MONTH_CATALOG_ENTRY,
    &DAY_CATALOG_ENTRY,
    &HOUR_CATALOG_ENTRY,
    &MINUTE_CATALOG_ENTRY,
    &NOW_CATALOG_ENTRY,
    &TODAY_CATALOG_ENTRY,
    &CLOCK_CATALOG_ENTRY,
];
