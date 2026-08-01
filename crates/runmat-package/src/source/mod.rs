mod catalog;
mod project;
mod symbols;

pub use catalog::{
    FrozenProject, FrozenSourceDescriptor, PackageMount, PackageSourceCatalog, ProjectRevision,
    SourceCatalog, StableSourceId,
};
pub use project::{
    build_frozen_project, build_frozen_project_async, discover_frozen_project_from,
    discover_frozen_project_from_async, FrozenProjectError,
};
pub use symbols::{
    discover_known_project_symbols_from_source_name,
    discover_known_project_symbols_from_source_name_async,
    discover_source_symbols_from_source_name, discover_source_symbols_from_source_name_async,
    source_symbols_from_index, DiscoverSourceSymbolsError, DiscoveredSourceSymbols,
    ProjectSymbolDefinition,
};
