//! Configuration system for RunMat
//!
//! File-based config discovery for RunMat:
//! 1. Explicit config path via RUNMAT_CONFIG
//! 2. Project config discovery (runmat.toml, runmat.json)
//! 3. User config (~/.config/runmat/config.toml|json)
//! 4. Built-in defaults

mod loader;
mod project;
mod schema;

pub use loader::ConfigLoader;
pub use project::{
    build_project_composition_graph, build_project_composition_graph_async,
    build_project_source_index, build_project_source_index_async,
    discover_known_project_symbols_from_source_name,
    discover_known_project_symbols_from_source_name_async, discover_project_manifest_from,
    discover_project_manifest_from_async, discover_project_symbols_from,
    discover_project_symbols_from_async, discover_project_symbols_from_source_name,
    discover_project_symbols_from_source_name_async, load_project_manifest,
    load_project_manifest_async, parse_project_manifest_json, parse_project_manifest_toml,
    resolve_named_entrypoint_from, resolve_project_entrypoint, resolve_project_source_input_from,
    DiscoverProjectEntrypointError, DiscoverProjectSymbolsError, DiscoveredProjectEntrypoint,
    DiscoveredProjectSymbols, ProjectCompositionError, ProjectCompositionGraph,
    ProjectCompositionPackage, ProjectDependency, ProjectEntrypoint, ProjectEntrypointResolveError,
    ProjectManifest, ProjectManifestLoadError, ProjectManifestValidationError, ProjectPackage,
    ProjectSourceFile, ProjectSourceIndex, ProjectSourceIndexError, ProjectSources,
    ResolveProjectSourceInputError, ResolvedEntrypointTarget, ResolvedProjectEntrypoint,
    PROJECT_MANIFEST_FILENAME, PROJECT_MANIFEST_FILENAMES,
};
pub use schema::{
    error_namespace_for_language_compat, AccelPowerPreference, AccelerateConfig,
    AccelerateProviderPreference, AutoOffloadConfig, AutoOffloadLogLevel, ExportConfig,
    ExportFormat, GcConfig, GcPreset, GuiConfig, JitConfig, JitOptLevel, LanguageCompatMode,
    LanguageConfig, LogLevel, LoggingConfig, PlotBackend, PlotMode, PlottingConfig, RunMatConfig,
    RuntimeConfig, TelemetryConfig, TelemetryDrainMode,
};
