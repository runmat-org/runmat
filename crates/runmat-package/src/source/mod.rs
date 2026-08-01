mod acquisition;
mod catalog;
mod git_policy;
mod handoff;
mod inventory;
mod path_project;
mod project_resolver;
mod registry_policy;
mod registry_release;
mod server_policy;
mod supply_chain;
mod symbols;
mod vendor;

pub use acquisition::{SourceAcquisitionIntent, SourceAcquisitionPolicy, SourceLockAction};
pub use catalog::{
    FrozenProject, FrozenSourceDescriptor, PackageMount, PackageSourceCatalog, ProjectRevision,
    SourceCatalog, StableSourceId, VisibleProjectSource,
};
pub use git_policy::{
    plan_git_acquisition, validate_git_acquisition, GitAcquisitionIntent, GitAcquisitionPlan,
    GitAcquisitionPolicy, GitLockAction, GitPolicyError,
};
pub use handoff::{
    FrozenProjectHandoff, FrozenProjectHandoffError, FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION,
};
pub use inventory::{SourceInventory, SourceInventoryEntry, SOURCE_INVENTORY_SCHEMA_VERSION};
pub use path_project::{
    build_frozen_project, build_frozen_project_async, discover_frozen_project_from,
    discover_frozen_project_from_async, FrozenProjectError, PathProjectError,
};
pub use project_resolver::{
    resolve_project_async, GitPackageMount, GitPackageProvider, PackageSourceProvider,
    ProjectResolveError, ProjectResolveOptions, RegistryPackageMount, ResolvedProject,
    ServerProjectPackageMount,
};
pub use registry_policy::{
    plan_registry_acquisition, plan_registry_candidates, plan_selected_registry_acquisition,
    validate_registry_acquisition, RegistryAcquisitionPlan, RegistryCandidatePlan,
    RegistryPolicyError,
};
pub use registry_release::{
    RegistryCandidateRecord, RegistryPackageReference, RegistryReleaseDependency,
    RegistryReleaseMetadata,
};
pub use server_policy::{
    plan_server_project_acquisition, validate_server_project_acquisition, ServerPolicyError,
    ServerProjectAcquisitionPlan, ServerSnapshotSelector,
};
pub use supply_chain::{
    BuildProvenance, PackageTrustTier, RegistryReleaseSupplyChain, SbomFormat, SbomReference,
    WrapperProvenance, RELEASE_SUPPLY_CHAIN_SCHEMA_VERSION,
};
pub use symbols::{
    discover_known_project_symbols_from_source_name,
    discover_known_project_symbols_from_source_name_async,
    discover_source_symbols_from_source_name, discover_source_symbols_from_source_name_async,
    source_symbols_from_frozen, source_symbols_from_index, DiscoverSourceSymbolsError,
    DiscoveredSourceSymbols, ProjectSymbolDefinition,
};
pub use vendor::{
    VendorManifest, VendoredPackage, VENDOR_MANIFEST_FILENAME, VENDOR_SCHEMA_VERSION,
};
