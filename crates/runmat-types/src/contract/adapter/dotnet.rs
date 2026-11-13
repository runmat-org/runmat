use super::{model::complete_sequence, *};

pub fn dotnet_contract() -> ForeignAdapterContract {
    use ForeignAdapterNonGoal as NonGoal;
    ForeignAdapterContract {
        schema_version: FOREIGN_ADAPTER_CONTRACT_SCHEMA_VERSION,
        adapter: PlannedForeignAdapter::DotNet.adapter_id().into(),
        contract_version: DOTNET_ADAPTER_CONTRACT_VERSION,
        delivery: ForeignAdapterDelivery::Planned,
        platforms: vec![
            ForeignAdapterPlatform::WindowsX86_64,
            ForeignAdapterPlatform::MacOsX86_64,
            ForeignAdapterPlatform::MacOsAarch64,
            ForeignAdapterPlatform::LinuxX86_64,
        ],
        surfaces: vec![
            ForeignAdapterSurface::Dotnetenv,
            ForeignAdapterSurface::NetAddAssembly,
            ForeignAdapterSurface::ManagedNamespaces,
            ForeignAdapterSurface::ManagedObjects,
            ForeignAdapterSurface::ManagedArrays,
            ForeignAdapterSurface::ManagedDelegates,
            ForeignAdapterSurface::ManagedEvents,
        ],
        value_mappings: vec![
            ForeignValueMapping::ScalarNumerics,
            ForeignValueMapping::FixedWidthIntegers,
            ForeignValueMapping::Logical,
            ForeignValueMapping::Complex,
            ForeignValueMapping::Utf16Text,
            ForeignValueMapping::DenseArrays,
            ForeignValueMapping::CellsAndSequences,
            ForeignValueMapping::StructsAndDictionaries,
            ForeignValueMapping::DateTimeAndDuration,
            ForeignValueMapping::NullAndMissing,
            ForeignValueMapping::ManagedArrays,
            ForeignValueMapping::OpaqueObjects,
        ],
        invocation: vec![
            ForeignInvocationModel::NamespaceAndTypeResolution,
            ForeignInvocationModel::ReflectionOverloadResolution,
            ForeignInvocationModel::ConstructorsMethodsAndProperties,
            ForeignInvocationModel::GenericTypeConstruction,
            ForeignInvocationModel::ByReferenceArguments,
            ForeignInvocationModel::OptionalAndNamedArguments,
        ],
        callbacks: vec![
            ForeignCallbackModel::ManagedDelegates,
            ForeignCallbackModel::ManagedEvents,
            ForeignCallbackModel::RuntimeContextReentry,
        ],
        threading: vec![
            ForeignThreadingModel::ManagedRuntimeThreads,
            ForeignThreadingModel::OriginRuntimeContext,
        ],
        lifecycle: vec![
            ForeignLifecycleModel::ProcessRuntime,
            ForeignLifecycleModel::SessionAssemblyContext,
            ForeignLifecycleModel::SessionObjectRegistry,
            ForeignLifecycleModel::GenerationFencedRestart,
            ForeignLifecycleModel::DeterministicShutdown,
        ],
        packaging: vec![
            ForeignPackagingModel::ExactRuntimeIdentity,
            ForeignPackagingModel::ManagedAssembly,
            ForeignPackagingModel::NativeDependency,
            ForeignPackagingModel::RuntimeConfig,
            ForeignPackagingModel::CanonicalArtifactIdentity,
        ],
        isolation: vec![
            ForeignIsolationModel::TrustedInProcess,
            ForeignIsolationModel::SameBinaryIsolatedHost,
            ForeignIsolationModel::AuthenticatedBoundedRpc,
            ForeignIsolationModel::NoObjectIdentityTransfer,
        ],
        sequencing: complete_sequence(),
        non_goals: vec![
            NonGoal::BrowserLocalRuntime,
            NonGoal::SilentRuntimeSubstitution,
            NonGoal::CrossProcessObjectIdentity,
            NonGoal::ArbitraryObjectPersistence,
            NonGoal::UncheckedPointerProjection,
            NonGoal::DesktopUiAutomation,
        ],
    }
}
