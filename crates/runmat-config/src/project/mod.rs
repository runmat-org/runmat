mod composition;
mod entrypoint;
mod manifest;
mod source_index;
mod symbols;

pub use composition::{
    build_project_composition_graph, build_project_composition_graph_async,
    ProjectCompositionError, ProjectCompositionGraph, ProjectCompositionPackage,
};
pub use entrypoint::{
    resolve_named_entrypoint_from, resolve_named_entrypoint_from_async, resolve_project_entrypoint,
    resolve_project_entrypoint_async, resolve_project_source_input_from,
    resolve_project_source_input_from_async, DiscoverProjectEntrypointError,
    DiscoveredProjectEntrypoint, ProjectEntrypointResolveError, ResolveProjectSourceInputError,
    ResolvedEntrypointTarget, ResolvedProjectEntrypoint,
};
pub use manifest::{
    discover_project_manifest_from, discover_project_manifest_from_async, load_project_manifest,
    load_project_manifest_async, parse_project_manifest_json, parse_project_manifest_toml,
    ProjectDependency, ProjectEntrypoint, ProjectManifest, ProjectManifestLoadError,
    ProjectManifestValidationError, ProjectPackage, ProjectSources, PROJECT_MANIFEST_FILENAME,
    PROJECT_MANIFEST_FILENAMES,
};
pub use source_index::{
    build_loose_source_index, build_loose_source_index_async, build_project_source_index,
    build_project_source_index_async, project_source_file_from_path, ProjectSourceFile,
    ProjectSourceIndex, ProjectSourceIndexError,
};
pub use symbols::{
    discover_known_project_symbols_from_source_name,
    discover_known_project_symbols_from_source_name_async, discover_project_symbols_from,
    discover_project_symbols_from_async, discover_project_symbols_from_source_name,
    discover_project_symbols_from_source_name_async, discover_source_symbols_from_source_name,
    discover_source_symbols_from_source_name_async, source_symbols_from_index,
    DiscoverProjectSymbolsError, DiscoverSourceSymbolsError, DiscoveredProjectSymbols,
    DiscoveredSourceSymbols, ProjectSymbolDefinition,
};
