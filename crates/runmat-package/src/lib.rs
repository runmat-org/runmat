//! Portable package-domain authority for RunMat.

pub mod error;
pub mod graph;
pub mod identity;
pub mod lock;
pub mod manifest;
pub mod policy;
pub mod resolve;
pub mod source;

pub use error::{GraphError, IdentityError, LockError, ManifestError, PackageError, ResolveError};
pub use graph::{
    build_path_graph, build_resolved_graph, DependencyPath, GraphEdge, GraphPackage, PackageGraph,
    PathGraphInput, PathPackageInput, ResolvedDependencyInput, ResolvedGraphInput,
    ResolvedPackageInput, VisibilityResolution,
};
pub use identity::{
    CanonicalPackageId, ContentDigest, DigestAlgorithm, GitCommitId, GitObjectAlgorithm,
    GitRepositoryUrl, GitSourceId, NormalizedRelativePath, PackageAlias, PackageInstanceId,
    PackageVersion, PathSourceId, RegistryId, RegistryOrigin, RegistryReleaseId, RegistrySourceId,
    ServerProjectSourceId, SourceId,
};
pub use lock::{
    decode_lock, diff_locks, encode_lock, reconcile_path_lock, LockCompatibility, LockDiff,
    LockSelection, LockedEdge, LockedPackage, PackageLock, PathLockDecision, PathLockMode,
    RootLock, LOCK_SCHEMA_VERSION, RESOLVER_FORMAT_VERSION,
};
pub use manifest::{
    DependencyGroup, DependencyLocator, DependencySpec, GitSelector, HostCapability,
    PackageManifest, PublicationDeclaration, RegistryDeclaration, SourceReplacement,
    TargetEnvironment, TargetPredicate,
};
pub use resolve::{
    acquire_candidates, acquire_candidates_with_policy, dependency_tree, plan_update, resolve, why,
    CandidateIndex, CandidateMetadata, CandidateProvider, CandidateQuery, Incompatibility,
    RequirementPath, Resolution, ResolutionEdge, ResolutionPackage, ResolutionRequest,
    ResolutionRequirement, SourceSelectionPolicy, UpdatePlan, UpdatePolicy,
};
pub use source::{
    build_frozen_project, build_frozen_project_async, discover_frozen_project_from,
    discover_frozen_project_from_async, discover_known_project_symbols_from_source_name,
    discover_known_project_symbols_from_source_name_async,
    discover_source_symbols_from_source_name, discover_source_symbols_from_source_name_async,
    plan_git_acquisition, plan_registry_acquisition, plan_registry_candidates,
    plan_selected_registry_acquisition, plan_server_project_acquisition, resolve_project_async,
    source_symbols_from_frozen, source_symbols_from_index, validate_git_acquisition,
    validate_registry_acquisition, validate_server_project_acquisition, ArtifactContentCipher,
    BuildProvenance, DiscoverSourceSymbolsError, DiscoveredSourceSymbols,
    EncryptedArtifactMetadata, FrozenProject, FrozenProjectError, FrozenProjectHandoff,
    FrozenProjectHandoffError, FrozenSourceDescriptor, GitAcquisitionIntent, GitAcquisitionPlan,
    GitAcquisitionPolicy, GitLockAction, GitPackageMount, GitPackageProvider, GitPolicyError,
    KeyEnvelopeAlgorithm, PackageKeyEnvelope, PackageMount, PackageSourceCatalog,
    PackageSourceProvider, PackageTrustTier, ProjectResolveError, ProjectResolveOptions,
    ProjectRevision, ProjectSymbolDefinition, RecipientEncryptionKey, RecipientKeyAlgorithm,
    RegistryAcquisitionPlan, RegistryCandidatePlan, RegistryCandidateRecord, RegistryPackageMount,
    RegistryPackageReference, RegistryPolicyError, RegistryReleaseDependency,
    RegistryReleaseMetadata, RegistryReleaseSupplyChain, ResolvedProject, SbomFormat,
    SbomReference, ServerPolicyError, ServerProjectAcquisitionPlan, ServerProjectPackageMount,
    ServerSnapshotSelector, SourceAcquisitionIntent, SourceAcquisitionPolicy, SourceCatalog,
    SourceInventory, SourceInventoryEntry, SourceLockAction, StableSourceId, VendorManifest,
    VendoredPackage, VisibleProjectSource, WrapperProvenance, AES_256_GCM_NONCE_BYTE_LEN,
    AES_256_GCM_WRAPPED_KEY_BYTE_LEN, ENCRYPTED_ARTIFACT_SCHEMA_VERSION,
    FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION, P256_PUBLIC_KEY_BYTE_LEN,
    PACKAGE_KEY_ENVELOPE_SCHEMA_VERSION, RELEASE_SUPPLY_CHAIN_SCHEMA_VERSION,
    SOURCE_INVENTORY_SCHEMA_VERSION, VENDOR_MANIFEST_FILENAME, VENDOR_SCHEMA_VERSION,
};
