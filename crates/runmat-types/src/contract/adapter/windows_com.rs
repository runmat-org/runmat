use super::{model::complete_sequence, *};

pub fn windows_com_contract() -> ForeignAdapterContract {
    use ForeignAdapterNonGoal as NonGoal;
    ForeignAdapterContract {
        schema_version: FOREIGN_ADAPTER_CONTRACT_SCHEMA_VERSION,
        adapter: PlannedForeignAdapter::WindowsCom.adapter_id().into(),
        contract_version: WINDOWS_COM_ADAPTER_CONTRACT_VERSION,
        delivery: ForeignAdapterDelivery::Planned,
        platforms: vec![ForeignAdapterPlatform::WindowsX86_64],
        surfaces: vec![
            ForeignAdapterSurface::Actxserver,
            ForeignAdapterSurface::AutomationInvoke,
            ForeignAdapterSurface::AutomationProperties,
            ForeignAdapterSurface::AutomationEvents,
            ForeignAdapterSurface::ExplicitRelease,
        ],
        value_mappings: vec![
            ForeignValueMapping::ScalarNumerics,
            ForeignValueMapping::FixedWidthIntegers,
            ForeignValueMapping::Logical,
            ForeignValueMapping::Utf16Text,
            ForeignValueMapping::DenseArrays,
            ForeignValueMapping::DateTimeAndDuration,
            ForeignValueMapping::NullAndMissing,
            ForeignValueMapping::ComVariant,
            ForeignValueMapping::ComSafeArray,
            ForeignValueMapping::OpaqueObjects,
        ],
        invocation: vec![
            ForeignInvocationModel::ConstructorsMethodsAndProperties,
            ForeignInvocationModel::ComDispatch,
            ForeignInvocationModel::ComTypeInformation,
            ForeignInvocationModel::ByReferenceArguments,
            ForeignInvocationModel::OptionalAndNamedArguments,
        ],
        callbacks: vec![
            ForeignCallbackModel::ComConnectionPoints,
            ForeignCallbackModel::RuntimeContextReentry,
        ],
        threading: vec![
            ForeignThreadingModel::OriginRuntimeContext,
            ForeignThreadingModel::ComSingleThreadedApartment,
            ForeignThreadingModel::ComMultiThreadedApartment,
            ForeignThreadingModel::ApartmentMarshalling,
        ],
        lifecycle: vec![
            ForeignLifecycleModel::SessionObjectRegistry,
            ForeignLifecycleModel::GenerationFencedRestart,
            ForeignLifecycleModel::ApartmentInitialization,
            ForeignLifecycleModel::ExplicitObjectRelease,
            ForeignLifecycleModel::DeterministicShutdown,
        ],
        packaging: vec![
            ForeignPackagingModel::ComTypeLibrary,
            ForeignPackagingModel::ComClassIdentity,
            ForeignPackagingModel::ExternalRegistrationPrerequisite,
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
            NonGoal::AutomaticComRegistration,
            NonGoal::CrossProcessObjectIdentity,
            NonGoal::ArbitraryObjectPersistence,
            NonGoal::UncheckedPointerProjection,
            NonGoal::DesktopUiAutomation,
        ],
    }
}
