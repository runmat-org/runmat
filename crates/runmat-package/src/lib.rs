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
    build_path_graph, DependencyPath, GraphEdge, GraphPackage, PackageGraph, PathGraphInput,
    PathPackageInput, VisibilityResolution,
};
pub use identity::{
    CanonicalPackageId, ContentDigest, DigestAlgorithm, GitCommitId, GitObjectAlgorithm,
    GitRepositoryUrl, GitSourceId, NormalizedRelativePath, PackageAlias, PackageInstanceId,
    PackageVersion, PathSourceId, RegistryId, RegistrySourceId, ServerProjectSourceId, SourceId,
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
    source_symbols_from_frozen, source_symbols_from_index, DiscoverSourceSymbolsError,
    DiscoveredSourceSymbols, FrozenProject, FrozenProjectError, FrozenProjectHandoff,
    FrozenProjectHandoffError, FrozenSourceDescriptor, PackageMount, PackageSourceCatalog,
    ProjectRevision, ProjectSymbolDefinition, SourceCatalog, StableSourceId, VisibleProjectSource,
    FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION,
};
