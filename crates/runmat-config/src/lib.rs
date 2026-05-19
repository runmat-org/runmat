//! Configuration system for RunMat
//!
//! Supports multiple configuration sources with proper precedence:
//! 1. Command-line arguments (highest priority)
//! 2. Environment variables
//! 3. Configuration files (.runmat.yaml, .runmat.json, etc.)
//! 4. Built-in defaults (lowest priority)

mod loader;
mod project;
mod schema;

pub use loader::ConfigLoader;
pub use project::{
    build_project_composition_graph, build_project_source_index, discover_project_manifest_from,
    discover_project_symbols_from, discover_project_symbols_from_source_name,
    load_project_manifest, parse_project_manifest_toml, resolve_named_entrypoint_from,
    resolve_project_entrypoint, resolve_project_source_input_from, DiscoverProjectEntrypointError,
    DiscoverProjectSymbolsError, DiscoveredProjectEntrypoint, DiscoveredProjectSymbols,
    ProjectCompositionError, ProjectCompositionGraph, ProjectCompositionPackage, ProjectDependency,
    ProjectEntrypoint, ProjectEntrypointResolveError, ProjectManifest, ProjectManifestLoadError,
    ProjectManifestValidationError, ProjectPackage, ProjectSourceFile, ProjectSourceIndex,
    ProjectSourceIndexError, ProjectSources, ResolveProjectSourceInputError,
    ResolvedEntrypointTarget, ResolvedProjectEntrypoint, PROJECT_MANIFEST_FILENAME,
};
pub use schema::{
    error_namespace_for_language_compat, AccelPowerPreference, AccelerateConfig,
    AccelerateProviderPreference, AutoOffloadConfig, AutoOffloadLogLevel, ExportConfig,
    ExportFormat, GcConfig, GcPreset, GuiConfig, JitConfig, JitOptLevel, JupyterConfig,
    JupyterOutputFormat, JupyterPerformanceConfig, JupyterStaticConfig, JupyterWidgetConfig,
    KernelConfig, KernelPorts, LanguageCompatMode, LanguageConfig, LogLevel, LoggingConfig,
    PlotBackend, PlotMode, PlottingConfig, RunMatConfig, RuntimeConfig, TelemetryConfig,
    TelemetryDrainMode,
};
