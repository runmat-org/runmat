//! Canonical calendar-duration identity contracts and compatibility extensions.
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &CALDAYS_CATALOG_ENTRY,
    &CALENDARDURATION_CATALOG_ENTRY,
    &CALMONTHS_CATALOG_ENTRY,
    &CALQUARTERS_CATALOG_ENTRY,
    &CALWEEKS_CATALOG_ENTRY,
    &CALYEARS_CATALOG_ENTRY,
    &ISCALENDARDURATION_CATALOG_ENTRY,
];

/// The original RunMat day-array shortcut and empty constructor are retained
/// for existing projects, but are not the documented matrix/component forms.
pub const CALENDAR_DURATION_SHORTHAND_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "calendar-duration-shorthand",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "The empty constructor, single-array-of-days, four/five-component, and calendarDuration copy forms are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:CalendarDurationShorthandExtension"),
    };

pub const CALENDAR_DURATION_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-resident-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Resident numeric calendar input is gathered to a host calendarDuration only in RunMat mode",
        error_identifier: Some("RunMat:compatibility:DatetimeGpuInputExtension"),
    };
pub const CALENDAR_DURATION_LOGICAL_COMPONENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "calendar-duration-logical-component",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Logical calendar constructor components are a RunMat extension; unit helper functions accept logical input in both modes",
        error_identifier: Some("RunMat:compatibility:CalendarDurationLogicalComponentExtension"),
    };

pub const CALENDAR_DURATION_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CALENDAR_DURATION_SHORTHAND_EXTENSION,
    CALENDAR_DURATION_GPU_INPUT_EXTENSION,
    CALENDAR_DURATION_LOGICAL_COMPONENT_EXTENSION,
];
pub const CALENDAR_UNIT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [CALENDAR_DURATION_GPU_INPUT_EXTENSION];

const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    predicate: bool,
    extensions: &'static [BuiltinExtensionDescriptor],
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "datetime/calendar-duration",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: if predicate {
                BuiltinAsyncBehavior::NeverSuspends
            } else {
                BuiltinAsyncBehavior::MaySuspend
            },
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: if predicate {
                &[]
            } else {
                &[EffectKind::MaySuspend, EffectKind::MayThrow]
            },
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: if predicate {
                BuiltinResidencyPolicy::Host
            } else {
                BuiltinResidencyPolicy::GatherToHost
            },
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

macro_rules! unit_entry {
    ($constant:ident, $name:literal, $descriptor:ident, $documentation:ident) => {
        pub const $constant: BuiltinCatalogEntry = entry(
            $name,
            &super::contracts::$descriptor,
            super::documentation::$documentation,
            super::inference::unit,
            false,
            &CALENDAR_UNIT_EXTENSIONS,
            &super::contracts::UNIT_INTEGER_CAPABILITIES,
        );
    };
}

unit_entry!(CALDAYS_CATALOG_ENTRY, "caldays", CALDAYS, CALDAYS);
unit_entry!(CALWEEKS_CATALOG_ENTRY, "calweeks", CALWEEKS, CALWEEKS);
unit_entry!(CALMONTHS_CATALOG_ENTRY, "calmonths", CALMONTHS, CALMONTHS);
unit_entry!(
    CALQUARTERS_CATALOG_ENTRY,
    "calquarters",
    CALQUARTERS,
    CALQUARTERS
);
unit_entry!(CALYEARS_CATALOG_ENTRY, "calyears", CALYEARS, CALYEARS);

pub const CALENDARDURATION_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "calendarDuration",
    &super::contracts::CALENDARDURATION,
    super::documentation::CALENDARDURATION,
    super::inference::constructor,
    false,
    &CALENDAR_DURATION_EXTENSIONS,
    &super::contracts::CONSTRUCTOR_INTEGER_CAPABILITIES,
);
pub const ISCALENDARDURATION_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "iscalendarduration",
    &super::contracts::ISCALENDARDURATION,
    super::documentation::ISCALENDARDURATION,
    super::inference::predicate,
    true,
    &[],
    &super::contracts::PREDICATE_INTEGER_CAPABILITIES,
);
