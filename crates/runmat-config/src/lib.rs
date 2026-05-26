//! Configuration system for RunMat
//!
//! Supports multiple configuration sources with proper precedence:
//! 1. Command-line arguments (highest priority)
//! 2. Explicit config path via RUNMAT_CONFIG
//! 3. Configuration files (runmat.toml, runmat.json)
//! 4. Built-in defaults (lowest priority)

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
