use super::{ForeignAdapterDelivery, FOREIGN_ADAPTER_CONTRACT_SCHEMA_VERSION};
use crate::{contract::schema::validate_token, SchemaValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignAdapterPlatform {
    WindowsX86_64,
    MacOsX86_64,
    MacOsAarch64,
    LinuxX86_64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignAdapterSurface {
    Dotnetenv,
    NetAddAssembly,
    ManagedNamespaces,
    ManagedObjects,
    ManagedArrays,
    ManagedDelegates,
    ManagedEvents,
    Actxserver,
    AutomationInvoke,
    AutomationProperties,
    AutomationEvents,
    ExplicitRelease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignValueMapping {
    ScalarNumerics,
    FixedWidthIntegers,
    Logical,
    Complex,
    Utf16Text,
    DenseArrays,
    CellsAndSequences,
    StructsAndDictionaries,
    DateTimeAndDuration,
    NullAndMissing,
    ManagedArrays,
    ComVariant,
    ComSafeArray,
    OpaqueObjects,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignInvocationModel {
    NamespaceAndTypeResolution,
    ReflectionOverloadResolution,
    ConstructorsMethodsAndProperties,
    GenericTypeConstruction,
    ComDispatch,
    ComTypeInformation,
    ByReferenceArguments,
    OptionalAndNamedArguments,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignCallbackModel {
    ManagedDelegates,
    ManagedEvents,
    ComConnectionPoints,
    RuntimeContextReentry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignThreadingModel {
    ManagedRuntimeThreads,
    OriginRuntimeContext,
    ComSingleThreadedApartment,
    ComMultiThreadedApartment,
    ApartmentMarshalling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignLifecycleModel {
    ProcessRuntime,
    SessionAssemblyContext,
    SessionObjectRegistry,
    GenerationFencedRestart,
    ApartmentInitialization,
    ExplicitObjectRelease,
    DeterministicShutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignPackagingModel {
    ExactRuntimeIdentity,
    ManagedAssembly,
    NativeDependency,
    RuntimeConfig,
    ComTypeLibrary,
    ComClassIdentity,
    ExternalRegistrationPrerequisite,
    CanonicalArtifactIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignIsolationModel {
    TrustedInProcess,
    SameBinaryIsolatedHost,
    AuthenticatedBoundedRpc,
    NoObjectIdentityTransfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignAdapterSequence {
    DiscoveryAndMetadata,
    ValuesAndIdentity,
    InvocationAndErrors,
    CallbacksAndEvents,
    PackagingAndIsolation,
    ProductQualification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignAdapterNonGoal {
    BrowserLocalRuntime,
    SilentRuntimeSubstitution,
    AutomaticComRegistration,
    CrossProcessObjectIdentity,
    ArbitraryObjectPersistence,
    UncheckedPointerProjection,
    DesktopUiAutomation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignAdapterContract {
    pub schema_version: u16,
    pub adapter: String,
    pub contract_version: u32,
    pub delivery: ForeignAdapterDelivery,
    pub platforms: Vec<ForeignAdapterPlatform>,
    pub surfaces: Vec<ForeignAdapterSurface>,
    pub value_mappings: Vec<ForeignValueMapping>,
    pub invocation: Vec<ForeignInvocationModel>,
    pub callbacks: Vec<ForeignCallbackModel>,
    pub threading: Vec<ForeignThreadingModel>,
    pub lifecycle: Vec<ForeignLifecycleModel>,
    pub packaging: Vec<ForeignPackagingModel>,
    pub isolation: Vec<ForeignIsolationModel>,
    pub sequencing: Vec<ForeignAdapterSequence>,
    pub non_goals: Vec<ForeignAdapterNonGoal>,
}

impl ForeignAdapterContract {
    pub fn validate(&self) -> Result<(), SchemaValidationError> {
        if self.schema_version != FOREIGN_ADAPTER_CONTRACT_SCHEMA_VERSION {
            return Err(SchemaValidationError::new(
                "adapter_contract.schema_version",
                format!(
                    "unsupported version {}; expected {}",
                    self.schema_version, FOREIGN_ADAPTER_CONTRACT_SCHEMA_VERSION
                ),
            ));
        }
        validate_token("adapter_contract.adapter", &self.adapter, 96)?;
        if self.contract_version == 0 {
            return Err(SchemaValidationError::new(
                "adapter_contract.contract_version",
                "version must be non-zero",
            ));
        }
        validate_set("adapter_contract.platforms", &self.platforms)?;
        validate_set("adapter_contract.surfaces", &self.surfaces)?;
        validate_set("adapter_contract.value_mappings", &self.value_mappings)?;
        validate_set("adapter_contract.invocation", &self.invocation)?;
        validate_set("adapter_contract.callbacks", &self.callbacks)?;
        validate_set("adapter_contract.threading", &self.threading)?;
        validate_set("adapter_contract.lifecycle", &self.lifecycle)?;
        validate_set("adapter_contract.packaging", &self.packaging)?;
        validate_set("adapter_contract.isolation", &self.isolation)?;
        validate_set("adapter_contract.sequencing", &self.sequencing)?;
        validate_set("adapter_contract.non_goals", &self.non_goals)?;
        Ok(())
    }
}

fn validate_set<T: Ord>(path: &'static str, values: &[T]) -> Result<(), SchemaValidationError> {
    if values.is_empty() {
        return Err(SchemaValidationError::new(path, "must not be empty"));
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SchemaValidationError::new(
            path,
            "entries must be sorted and unique",
        ));
    }
    Ok(())
}

pub(super) fn complete_sequence() -> Vec<ForeignAdapterSequence> {
    vec![
        ForeignAdapterSequence::DiscoveryAndMetadata,
        ForeignAdapterSequence::ValuesAndIdentity,
        ForeignAdapterSequence::InvocationAndErrors,
        ForeignAdapterSequence::CallbacksAndEvents,
        ForeignAdapterSequence::PackagingAndIsolation,
        ForeignAdapterSequence::ProductQualification,
    ]
}
