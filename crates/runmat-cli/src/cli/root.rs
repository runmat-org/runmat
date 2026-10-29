use clap::parser::ValueSource;
use clap::{ArgMatches, Parser, Subcommand};
use runmat_config::runtime::{PlotBackend, PlotMode};
use runmat_server_client::auth::CredentialStoreMode;
use std::path::PathBuf;
use uuid::Uuid;

use crate::cli::parse::{parse_bool_env, parse_figure_size, parse_log_level_env};
use crate::cli::remote::{FsCommand, OrgCommand, ProjectCommand, RemoteCommand};
use crate::cli::value_types::{
    AotOptLevel, AotPolicy, CaptureFiguresMode, FigureSize, GcPreset, LogLevel,
    MeshElementOrderArg, MeshSourceUnitsArg, OptLevel,
};
use crate::cli::ColorMode;
use crate::cli::TestArgs;
use crate::cli::{
    BatchCommand, ClusterCommand, ExecutionTargetArgs, FeaRootArgs, JobCommand, MexArgs,
    NativeInterfaceCommand, NodeArgs, PackageCommand,
};

#[derive(Parser, Clone)]
#[command(
    name = "runmat",
    version = env!("CARGO_PKG_VERSION"),
    about = "High-performance MATLAB/Octave code runtime",
    long_about = r#"
RunMat is a modern, high-performance runtime for MATLAB/Octave.

It is built in Rust, and features a V8-inspired tiered execution model with a 
baseline interpreter feeding an optimizing JIT compiler built on Cranelift.

Key features:
• JIT compilation with Cranelift for optimal performance
• Generational garbage collection with configurable policies
• High-performance BLAS/LAPACK operations
• World-class error messages and debugging
• Compatible with MATLAB/Octave syntax and semantics

Performance Features:
• Multi-tier execution: interpreter + JIT compiler
• Adaptive optimization based on hotspot profiling
• Generational GC with write barriers and concurrent collection
• SIMD-optimized mathematical operations
• Zero-copy memory management where possible

Examples:
  runmat                                   # Start interactive REPL with JIT
  runmat --no-jit                          # Start REPL with interpreter only
  runmat --gc-preset low-latency           # Optimize GC for low latency
  runmat script.m                          # Execute MATLAB/Octave script
  runmat script.m --cluster engineering --trust-identity <fingerprint>
                                           # Execute on a trusted remote cluster
  runmat < script.m                        # Execute stdin as one script
  runmat -                                 # Explicit stdin script alias
  runmat --emit-bytecode script.m           # Emit bytecode disassembly
  runmat version --detailed                # Show detailed version information
"#,
    after_help = r#"
Environment Variables:
  RUNMAT_CONFIG=<path>        Explicit path to runmat.toml/runmat.json
  RUNMAT_API_KEY=<token>      Remote API token (remote commands)
  RUNMAT_SERVER_URL=<url>     Remote server URL (remote commands)
  RUNMAT_ORG_ID=<uuid>        Remote org override (remote commands)
  RUNMAT_PROJECT_ID=<uuid>    Remote project override (remote commands)
  RUNMAT_PACKAGE_CACHE_DIR=<path>  Shared package cache override
  NO_COLOR=<non-empty>        Disable ANSI color by default
  CLICOLOR=0                  Disable automatic ANSI color
  CLICOLOR_FORCE=<non-zero>   Force ANSI color for human output
  FORCE_COLOR=<1..3>          Force a specific color level

For more information, visit: https://github.com/runmat-org/runmat
"#
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Control ANSI styling for human-readable output
    #[arg(long, value_enum, default_value = "auto", global = true)]
    pub color: ColorMode,

    /// Resolve only from locally available package content
    #[arg(long, global = true)]
    pub offline: bool,

    /// Require an up-to-date lockfile
    #[arg(long, global = true)]
    pub locked: bool,

    /// Require the lockfile and prohibit network access or lock mutation
    #[arg(long, global = true)]
    pub frozen: bool,

    /// Enable debug logging
    #[arg(short, long, value_parser = parse_bool_env)]
    pub debug: bool,

    /// Set log level
    #[arg(long, value_enum, default_value = "warn", value_parser = parse_log_level_env)]
    pub log_level: LogLevel,

    /// Maximum number of call stack frames to record
    #[arg(long, default_value = "200")]
    pub callstack_limit: usize,

    /// Emit bytecode disassembly for a script (stdout if omitted path)
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "-")]
    pub emit_bytecode: Option<PathBuf>,

    /// Error identifier namespace prefix
    #[arg(long)]
    pub error_namespace: Option<String>,

    /// Configuration file path
    #[arg(long, env = "RUNMAT_CONFIG")]
    pub config: Option<PathBuf>,

    /// Disable native JIT compilation
    #[arg(long, value_parser = parse_bool_env)]
    pub no_jit: bool,

    /// Accepted threshold setting (does not control native tiering)
    #[arg(long, default_value = "10")]
    pub jit_threshold: u32,

    /// Accepted optimization setting (not applied to the native compiler)
    #[arg(long, value_enum, default_value = "speed")]
    pub jit_opt_level: OptLevel,

    /// GC configuration preset
    #[arg(long, value_enum)]
    pub gc_preset: Option<GcPreset>,

    /// Young generation size in MB
    #[arg(long)]
    pub gc_young_size: Option<usize>,

    /// Maximum number of GC threads
    #[arg(long)]
    pub gc_threads: Option<usize>,

    /// Enable GC statistics collection
    #[arg(long, value_parser = parse_bool_env)]
    pub gc_stats: bool,

    /// Verbose output for REPL and execution
    #[arg(short, long)]
    pub verbose: bool,

    /// Plotting mode
    #[arg(long, value_enum)]
    pub plot_mode: Option<PlotMode>,

    /// Force headless plotting mode
    #[arg(long, value_parser = parse_bool_env)]
    pub plot_headless: bool,

    /// Plotting backend
    #[arg(long, value_enum)]
    pub plot_backend: Option<PlotBackend>,

    /// Override scatter target points for GPU decimation
    #[arg(long)]
    pub plot_scatter_target: Option<u32>,

    /// Override surface vertex budget for GPU LOD
    #[arg(long)]
    pub plot_surface_vertex_budget: Option<u64>,

    /// Directory where run artifacts are written
    #[arg(long)]
    pub artifacts_dir: Option<PathBuf>,

    /// Path to write artifact manifest JSON
    #[arg(long)]
    pub artifacts_manifest: Option<PathBuf>,

    /// Figure capture mode when artifact output is enabled
    #[arg(long, value_enum, default_value = "auto")]
    pub capture_figures: CaptureFiguresMode,

    /// Figure export size (WIDTHxHEIGHT)
    #[arg(long, default_value = "1280x720", value_parser = parse_figure_size)]
    pub figure_size: FigureSize,

    /// Maximum number of figures to export
    #[arg(long, default_value = "8")]
    pub max_figures: usize,

    /// Generate sample configuration file
    #[arg(long)]
    pub generate_config: bool,

    /// Command to execute
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// MATLAB script file to execute (alternative to subcommands)
    pub script: Option<PathBuf>,

    #[command(flatten)]
    pub execution: ExecutionTargetArgs,

    /// Emit structured JSON for a study or remote run using the filename shorthand
    #[arg(long = "json", requires = "script")]
    pub script_json: bool,

    /// Select a declared function for remote execution using the filename shorthand
    #[arg(long = "function", requires_all = ["script", "cluster"])]
    pub script_function: Option<String>,

    /// Positional function inputs after -- when using the filename shorthand
    #[arg(last = true, requires = "script")]
    pub script_args: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CliOverrideSources {
    pub debug: bool,
    pub log_level: bool,
    pub callstack_limit: bool,
    pub jit_threshold: bool,
    pub jit_opt_level: bool,
    pub gc_stats: bool,
    pub verbose: bool,
}

impl CliOverrideSources {
    pub fn from_matches(matches: &ArgMatches) -> Self {
        Self {
            debug: Self::was_provided(matches, "debug"),
            log_level: Self::was_provided(matches, "log_level"),
            callstack_limit: Self::was_provided(matches, "callstack_limit"),
            jit_threshold: Self::was_provided(matches, "jit_threshold"),
            jit_opt_level: Self::was_provided(matches, "jit_opt_level"),
            gc_stats: Self::was_provided(matches, "gc_stats"),
            verbose: Self::was_provided(matches, "verbose"),
        }
    }

    fn was_provided(matches: &ArgMatches, id: &str) -> bool {
        matches
            .value_source(id)
            .is_some_and(|source| source != ValueSource::DefaultValue)
    }
}

#[derive(Subcommand, Clone)]
pub enum Commands {
    /// Start interactive REPL
    Repl {
        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run a MATLAB script or a .fea study or sweep
    Run {
        /// .m script or .fea document to run
        file: PathBuf,
        #[command(flatten)]
        root: FeaRootArgs,
        #[command(flatten)]
        execution: ExecutionTargetArgs,
        /// Select a declared function for remote execution
        #[arg(long, requires = "cluster")]
        function: Option<String>,
        /// Emit structured JSON for studies, sweeps and remote runs
        #[arg(long)]
        json: bool,
        /// Positional inputs; scripts currently reject them rather than inventing argv semantics
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Generate a validated solver mesh and factual meshing evidence from exact CAD
    Mesh {
        /// STEP, IGES, or B-rep source geometry
        source: PathBuf,
        #[command(flatten)]
        execution: ExecutionTargetArgs,
        /// Coordinate units used by source geometry that does not declare its units
        #[arg(long, value_enum, default_value = "meter")]
        source_units: MeshSourceUnitsArg,
        /// Canonical solver-mesh output path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Canonical meshing-evidence output path
        #[arg(long)]
        evidence: Option<PathBuf>,
        /// Global target edge length in meters
        #[arg(long, default_value_t = 0.01)]
        target_size: f64,
        /// Maximum curve and surface chordal deviation in meters
        #[arg(long, default_value_t = 0.0001)]
        deviation: f64,
        /// Tetrahedral element order
        #[arg(long, value_enum, default_value = "tet4")]
        element_order: MeshElementOrderArg,
        /// Hard maximum generated element count
        #[arg(long, default_value_t = 10_000_000)]
        max_elements: u64,
        /// Deterministic meshing seed
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Join touching bodies so coincident faces become shared, conformally meshed interfaces
        #[arg(long)]
        shared_interfaces: bool,
        /// Replace existing output files atomically
        #[arg(long)]
        force: bool,
        /// Emit a machine-readable result summary
        #[arg(long)]
        json: bool,
    },
    /// Compile a MATLAB program into a standalone native executable
    Compile {
        /// MATLAB script or function entrypoint
        file: PathBuf,
        /// Output executable path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Native optimization policy
        #[arg(long, value_enum, default_value = "speed")]
        optimization: AotOptLevel,
        /// Native runtime composition policy
        #[arg(long, value_enum, default_value = "native-specialized")]
        policy: AotPolicy,
        /// Explain why program symbols and runtime families are retained
        #[arg(long)]
        explain_link: bool,
        /// Write the deterministic reachability and link plan as JSON
        #[arg(long, value_name = "PATH")]
        link_plan_json: Option<PathBuf>,
        /// Explicit system linker driver
        #[arg(long)]
        linker: Option<PathBuf>,
        /// Canonical selected target-pack manifest (requires separate host admission)
        #[arg(long, requires_all = ["selected_components", "selected_host_admission"], conflicts_with = "linker")]
        selected_pack_manifest: Option<PathBuf>,
        /// Directory of raw target-pack components, addressed by component ID
        #[arg(long, requires_all = ["selected_pack_manifest", "selected_host_admission"])]
        selected_components: Option<PathBuf>,
        /// Independent operator-issued target, capability and qualified-link admission
        #[arg(long, requires_all = ["selected_pack_manifest", "selected_components"])]
        selected_host_admission: Option<PathBuf>,
        /// Retain temporary object, archive, and response-file inputs
        #[arg(long)]
        keep_temps: bool,
        /// Replace an existing output after preserving it until publication succeeds
        #[arg(short, long)]
        force: bool,
    },
    /// Build C, C++, Fortran, or CUDA sources into a platform-native MEX module
    Mex(MexArgs),
    /// Build CUDA sources into a GPU MEX module with nvcc
    Mexcuda(MexArgs),
    /// Prepare and inspect native shared-library interfaces
    NativeInterface {
        #[command(subcommand)]
        command: NativeInterfaceCommand,
    },
    /// Check a MATLAB script or .fea document without running it
    Check {
        /// .m or .fea file to check
        file: PathBuf,
        #[command(flatten)]
        root: FeaRootArgs,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
        /// Print completed analysis domains as well as diagnostics
        #[arg(short, long)]
        verbose: bool,
        /// Treat a diagnostic level as an error (currently: warnings)
        #[arg(short = 'D', value_name = "LEVEL")]
        deny: Vec<String>,
        /// Add a MATLAB lookup root while checking this source
        #[arg(long = "path", value_name = "DIRECTORY")]
        search_paths: Vec<PathBuf>,
    },
    /// Print the logical phases and analysis definition for one .fea study or sweep
    Plan {
        /// Root .fea document
        file: PathBuf,
        #[command(flatten)]
        root: FeaRootArgs,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
    /// Run one .fea study and capture portable scenes and previews for an example package
    Snapshot {
        /// Root .fea document in the example project
        file: PathBuf,
        /// Select one study declared by the root module
        #[arg(long)]
        study: Option<String>,
        /// Snapshot metadata output; scenes and previews are written beside it
        #[arg(long)]
        out: PathBuf,
    },
    /// Render an original saved numerical result without meshing or solving
    #[command(hide = true)]
    RenderSnapshot {
        snapshot: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify a documentation project's original saved field pages without execution
    #[command(hide = true)]
    VerifySnapshot {
        /// Original snapshot metadata, relative to the project working directory
        snapshot: PathBuf,
        /// Cold field replay receipt
        #[arg(long)]
        out: PathBuf,
    },
    /// Evaluate checked .fea criteria against an immutable saved result without solving again
    Reassess {
        /// Checked .fea source containing the revised assessment declarations
        file: PathBuf,
        /// Select one study declared by the root module
        #[arg(long)]
        study: Option<String>,
        /// Original saved study definition (sha256: followed by 64 lowercase hex digits)
        #[arg(long)]
        original_definition: runmat_execution::Digest,
        /// Content-addressed whole-study result manifest
        #[arg(long)]
        result_manifest: PathBuf,
        /// Emit the typed reassessment publication as JSON
        #[arg(long)]
        json: bool,
    },
    /// Show version information
    Version {
        /// Show detailed version information
        #[arg(long)]
        detailed: bool,
    },
    /// Show system information
    Info {
        /// Emit versioned, machine-readable compiled runtime feature facts
        #[arg(long)]
        json: bool,
    },
    /// Show acceleration provider information
    AccelInfo {
        /// Output provider information and telemetry as JSON
        #[arg(long)]
        json: bool,
        /// Reset provider telemetry counters after printing
        #[arg(long)]
        reset: bool,
    },
    /// Refresh authoritative calibration or inspect imported calibration data
    #[cfg(feature = "wgpu")]
    AccelCalibrate {
        /// Optional original-template input or diagnostic suite results JSON
        #[arg(required_unless_present = "refresh", conflicts_with = "refresh")]
        input: Option<PathBuf>,
        /// Measure current original CPU templates using the existing calibration owner
        #[arg(long)]
        refresh: bool,
        /// Preview updates without persisting the calibration cache
        #[arg(long)]
        dry_run: bool,
        /// Emit calibration outcome as JSON
        #[arg(long)]
        json: bool,
    },
    /// Garbage collection utilities
    Gc {
        #[command(subcommand)]
        gc_command: GcCommand,
    },
    /// Performance benchmarking
    Benchmark {
        /// Script file to benchmark
        file: PathBuf,
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u32,
        /// Enable JIT for benchmark
        #[arg(long)]
        jit: bool,
        /// Untimed warm-up invocations (report mode records them separately)
        #[arg(long, default_value_t = 3)]
        warmup: u32,
        /// Create a new JSONL report including cold-first and raw warm samples
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Configuration management
    Config {
        #[command(subcommand)]
        config_command: ConfigCommand,
    },
    /// Run against remote filesystem
    Remote {
        #[command(subcommand)]
        remote_command: RemoteCommand,
    },
    /// Authenticate with RunMat server
    Login {
        /// Server URL
        #[arg(long)]
        server: Option<String>,
        /// API key or access token
        #[arg(long)]
        api_key: Option<String>,
        /// Email for interactive login
        #[arg(long)]
        email: Option<String>,
        /// Credential storage mode: auto, secure, file, memory
        #[arg(long, default_value = "file")]
        credential_store: CredentialStoreMode,
        /// Default organization id
        #[arg(long, value_parser = runmat_server_client::auth::parse_org_id_input)]
        org: Option<Uuid>,
        /// Default project id
        #[arg(long, value_parser = runmat_server_client::auth::parse_project_id_input)]
        project: Option<Uuid>,
    },
    /// Organization management
    Org {
        #[command(subcommand)]
        org_command: OrgCommand,
    },
    /// Project management
    Project {
        #[command(subcommand)]
        project_command: ProjectCommand,
    },
    /// Remote filesystem commands
    Fs {
        #[command(subcommand)]
        fs_command: FsCommand,
    },
    /// Package resolution and shared-cache operations
    Package {
        #[command(subcommand)]
        package_command: PackageCommand,
    },
    /// Discover and run MATLAB-compatible tests
    Test(Box<TestArgs>),
    /// Submit and manage durable local batch jobs
    Batch {
        #[command(subcommand)]
        batch_command: BatchCommand,
    },
    /// Manage remote execution clusters and nodes
    Cluster {
        #[command(subcommand)]
        cluster_command: ClusterCommand,
    },
    /// Inspect or manage this machine's execution-node service
    Node(NodeArgs),
    /// Observe and manage durable encrypted remote jobs
    Job {
        #[command(subcommand)]
        job_command: JobCommand,
    },
    /// Create offline organization recovery keys and decrypt run results
    RecoveryKey {
        #[command(subcommand)]
        command: super::JobRecoveryCommand,
    },
}

#[derive(Subcommand, Clone)]
pub enum GcCommand {
    /// Show GC statistics
    Stats,
    /// Force minor collection
    Minor,
    /// Force major collection
    Major,
    /// Show current configuration
    Config,
    /// Test GC under stress
    Stress {
        /// Number of allocations
        #[arg(short, long, default_value = "10000")]
        allocations: usize,
    },
}

#[derive(Subcommand, Clone)]
pub enum ConfigCommand {
    /// Show resolved runtime configuration
    Show {
        /// Output format
        #[arg(long, value_enum, default_value = "toml")]
        format: ConfigFormat,
        /// Include exact per-field provenance (JSON output)
        #[arg(long)]
        provenance: bool,
    },
    /// Generate a starter runmat config file (project + runtime sections)
    Generate {
        /// Output file path
        #[arg(short, long, default_value = "runmat.toml")]
        output: PathBuf,
        /// Output format (overrides file extension when set)
        #[arg(long, value_enum)]
        format: Option<ConfigFormat>,
    },
    /// Validate configuration file
    Validate {
        /// Config file to validate
        config_file: PathBuf,
    },
    /// Show configuration file locations
    Paths,
    /// Read or edit only user settings, preserving other application settings
    User {
        #[command(subcommand)]
        command: ConfigUserCommand,
    },
}

#[derive(Subcommand, Clone)]
pub enum ConfigUserCommand {
    Show {
        #[arg(long)]
        file: Option<PathBuf>,
    },
    Set {
        /// Runtime-relative path segments, e.g. jit enabled
        #[arg(required = true, num_args = 1..)]
        field: Vec<String>,
        /// Typed JSON value, e.g. false, 4, or '"speed"'
        #[arg(long)]
        value: String,
        #[arg(long)]
        file: Option<PathBuf>,
    },
    Reset {
        #[arg(required = true, num_args = 1..)]
        field: Vec<String>,
        #[arg(long)]
        file: Option<PathBuf>,
    },
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Cli, Commands};
    use crate::cli::{
        AotPolicy, BatchCommand, ClusterCommand, ClusterStateArg, MeshSourceUnitsArg,
    };
    use clap::Parser;

    #[test]
    fn filename_and_run_share_remote_options_and_function_inputs() {
        for file in ["main.m", "model.fea"] {
            let options = [
                file,
                "--cluster",
                "engineering",
                "--workers",
                "4",
                "--trust-identity",
                "sha256:trusted-node",
                "--detach",
                "--json",
                "--",
                "2",
            ];
            let shorthand =
                Cli::try_parse_from(["runmat"].into_iter().chain(options.iter().copied())).unwrap();
            let explicit =
                Cli::try_parse_from(["runmat", "run"].into_iter().chain(options.iter().copied()))
                    .unwrap();
            let Some(Commands::Run {
                file: explicit_file,
                execution,
                function,
                json,
                args,
            }) = explicit.command
            else {
                panic!("expected explicit run command");
            };
            assert_eq!(shorthand.script.as_ref(), Some(&explicit_file));
            assert_eq!(shorthand.execution, execution);
            assert_eq!(shorthand.script_function, function);
            assert_eq!(shorthand.script_json, json);
            assert_eq!(shorthand.script_args, args);
            assert_eq!(
                shorthand.execution.resolve().unwrap(),
                execution.resolve().unwrap()
            );
        }
        let options = [
            "evaluate.m",
            "--function",
            "evaluate",
            "--cluster",
            "engineering",
            "--trust-identity",
            "sha256:trusted-node",
            "--",
            "case-a",
        ];
        let shorthand = Cli::try_parse_from(["runmat"].into_iter().chain(options)).unwrap();
        let explicit = Cli::try_parse_from(["runmat", "run"].into_iter().chain(options)).unwrap();
        let Some(Commands::Run { function, args, .. }) = explicit.command else {
            panic!("expected explicit function run");
        };
        assert_eq!(shorthand.script_function.as_deref(), Some("evaluate"));
        assert_eq!(shorthand.script_function, function);
        assert_eq!(shorthand.script_args, args);
        assert!(Cli::try_parse_from(["runmat", "evaluate.m", "--function", "evaluate"]).is_err());
        assert!(Cli::try_parse_from(["runmat", "job", "submit", "evaluate.m"]).is_err());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn calibration_refresh_and_input_are_exclusive_and_dry_run_is_preserved() {
        let cli =
            Cli::try_parse_from(["runmat", "accel-calibrate", "--refresh", "--dry-run"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::AccelCalibrate {
                input: None,
                refresh: true,
                dry_run: true,
                ..
            })
        ));
        assert!(
            Cli::try_parse_from(["runmat", "accel-calibrate", "results.json", "--refresh"])
                .is_err()
        );
        assert!(Cli::try_parse_from(["runmat", "accel-calibrate"]).is_err());
        assert!(
            Cli::try_parse_from(["runmat", "accel-calibrate", "results.json", "--dry-run"]).is_ok()
        );
    }

    #[test]
    fn public_namespace_ids_parse_and_foreign_namespaces_are_rejected() {
        let org = "org_0000000000000000000000000000002a";
        let project = "proj_0000000000000000000000000000002a";
        let parsed = Cli::try_parse_from(["runmat", "cluster", "list", "--org", org]).unwrap();
        let Some(Commands::Cluster {
            cluster_command: ClusterCommand::List { org: Some(id), .. },
        }) = parsed.command
        else {
            panic!("expected cluster list with organization");
        };
        assert_eq!(id, uuid::Uuid::from_u128(42));
        assert!(Cli::try_parse_from(["runmat", "cluster", "list", "--org", project]).is_err());
        assert!(Cli::try_parse_from(["runmat", "run", "main.fea", "--project", project]).is_ok());
        assert!(Cli::try_parse_from(["runmat", "run", "main.fea", "--project", org]).is_err());
    }

    #[test]
    fn removed_startup_snapshot_flag_is_rejected() {
        let error = match Cli::try_parse_from(["runmat", "--snapshot", "stdlib.snapshot", "main.m"])
        {
            Ok(_) => panic!("the removed startup snapshot flag must not parse"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("--snapshot"));
    }

    #[test]
    fn snapshot_is_no_longer_a_cli_command() {
        let cli = Cli::try_parse_from(["runmat", "snapshot"])
            .expect("ordinary positional input must continue to parse as a script path");
        assert!(cli.command.is_none());
        assert_eq!(
            cli.script.as_deref(),
            Some(std::path::Path::new("snapshot"))
        );
    }

    #[test]
    fn mesh_rejects_analysis_material_assignment() {
        let error = match Cli::try_parse_from([
            "runmat",
            "mesh",
            "part.step",
            "--material",
            "aluminum_6061",
        ]) {
            Ok(_) => panic!("meshing must not accept analysis material assignment"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("--material"));
    }

    #[test]
    fn mesh_accepts_explicit_source_units() {
        let cli = Cli::try_parse_from([
            "runmat",
            "mesh",
            "part.step",
            "--source-units",
            "millimeter",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Mesh {
                source_units: MeshSourceUnitsArg::Millimeter,
                ..
            })
        ));
    }

    #[test]
    fn mesh_uses_the_shared_remote_placement_surface() {
        let cli = Cli::try_parse_from([
            "runmat",
            "mesh",
            "part.step",
            "--cluster",
            "engineering",
            "--queue",
            "cpu",
            "--workers",
            "16",
            "--trust-identity",
            "sha256:trusted-node",
            "--output",
            "artifacts/part.mesh.cbor",
            "--evidence",
            "artifacts/part.evidence.cbor",
            "--force",
            "--detach",
        ])
        .unwrap();
        let Some(Commands::Mesh {
            output,
            evidence,
            force,
            execution,
            ..
        }) = cli.command
        else {
            panic!("expected mesh command")
        };
        assert_eq!(output.unwrap(), PathBuf::from("artifacts/part.mesh.cbor"));
        assert_eq!(
            evidence.unwrap(),
            PathBuf::from("artifacts/part.evidence.cbor")
        );
        assert!(force);
        assert_eq!(
            execution.resolve().unwrap(),
            crate::cli::ExecutionTarget::Remote(crate::cli::RemoteExecutionTarget {
                cluster: "engineering".into(),
                project: None,
                queue: "cpu".into(),
                trust_identity: "sha256:trusted-node".into(),
                idempotency_key: None,
                workers: 16,
                detach: true,
            })
        );
    }

    #[test]
    fn fea_run_uses_the_shared_remote_placement_surface() {
        let cli = Cli::try_parse_from([
            "runmat",
            "run",
            "model.fea",
            "--cluster",
            "engineering",
            "--queue",
            "cpu",
            "--workers",
            "8",
            "--trust-identity",
            "sha256:trusted-node",
            "--detach",
        ])
        .unwrap();
        let Some(Commands::Run { execution, .. }) = cli.command else {
            panic!("expected run command")
        };
        assert_eq!(
            execution.resolve().unwrap(),
            crate::cli::ExecutionTarget::Remote(crate::cli::RemoteExecutionTarget {
                cluster: "engineering".into(),
                project: None,
                queue: "cpu".into(),
                trust_identity: "sha256:trusted-node".into(),
                idempotency_key: None,
                workers: 8,
                detach: true,
            })
        );
    }

    #[test]
    fn typed_fea_reassessment_requires_an_original_publication_identity() {
        let digest = runmat_execution::Digest::sha256("original-study");
        let encoded = digest.to_string();
        let cli = Cli::try_parse_from([
            "runmat",
            "reassess",
            "revised.fea",
            "--study",
            "response",
            "--original-definition",
            &encoded,
            "--result-manifest",
            "saved.json",
            "--json",
        ])
        .unwrap();
        let Some(Commands::Reassess {
            original_definition,
            result_manifest,
            study,
            json,
            ..
        }) = cli.command
        else {
            panic!("expected reassessment command")
        };
        assert_eq!(original_definition, digest);
        assert_eq!(result_manifest, PathBuf::from("saved.json"));
        assert_eq!(study.as_deref(), Some("response"));
        assert!(json);
        assert!(Cli::try_parse_from([
            "runmat",
            "reassess",
            "revised.fea",
            "--original-definition",
            "not-a-digest",
            "--result-manifest",
            "saved.json",
        ])
        .is_err());
    }

    #[test]
    fn run_selects_one_fea_root_and_reuses_shared_placement() {
        let cli = Cli::try_parse_from([
            "runmat",
            "run",
            "model.fea",
            "--study",
            "static_case",
            "--cluster",
            "analysis-cluster",
            "--workers",
            "4",
            "--json",
        ])
        .unwrap();
        let Some(Commands::Run {
            file,
            root,
            execution,
            json,
            ..
        }) = cli.command
        else {
            panic!("expected run command");
        };
        assert_eq!(file, PathBuf::from("model.fea"));
        assert_eq!(root.study.as_deref(), Some("static_case"));
        assert!(root.sweep.is_none());
        assert_eq!(execution.cluster.as_deref(), Some("analysis-cluster"));
        assert_eq!(execution.workers, Some(4));
        assert!(json);

        for command in ["check", "plan", "run"] {
            assert!(Cli::try_parse_from([
                "runmat", command, "model.fea", "--study", "a", "--sweep", "b",
            ])
            .is_err());
        }
        let Some(Commands::Check { root, .. }) =
            Cli::try_parse_from(["runmat", "check", "model.fea", "--sweep", "cases"])
                .unwrap()
                .command
        else {
            panic!("expected check command");
        };
        assert_eq!(root.sweep.as_deref(), Some("cases"));
        let Some(Commands::Plan { root, json, .. }) =
            Cli::try_parse_from(["runmat", "plan", "model.fea", "--study", "a", "--json"])
                .unwrap()
                .command
        else {
            panic!("expected plan command");
        };
        assert_eq!(root.study.as_deref(), Some("a"));
        assert!(json);
    }

    #[test]
    fn run_argument_delimiter_keeps_target_like_text_in_the_workload() {
        let cli = Cli::try_parse_from([
            "runmat",
            "run",
            "model.m",
            "--",
            "--cluster",
            "not-placement",
        ])
        .unwrap();
        let Some(Commands::Run { args, .. }) = cli.command else {
            panic!("expected run command")
        };
        assert_eq!(args, ["--cluster", "not-placement"]);
    }

    #[test]
    fn compile_policy_defaults_to_the_supported_native_profile() {
        let cli = Cli::try_parse_from(["runmat", "compile", "main.m"]).unwrap();
        let Some(Commands::Compile { policy, .. }) = cli.command else {
            panic!("expected compile command");
        };
        assert_eq!(policy, AotPolicy::NativeSpecialized);
    }

    #[test]
    fn compile_policy_names_every_declared_composition_mode() {
        for (name, expected) in [
            ("native-specialized", AotPolicy::NativeSpecialized),
            ("closed-world", AotPolicy::ClosedWorld),
            ("dynamic-runtime", AotPolicy::DynamicRuntime),
            ("portable", AotPolicy::Portable),
        ] {
            let cli =
                Cli::try_parse_from(["runmat", "compile", "main.m", "--policy", name]).unwrap();
            let Some(Commands::Compile { policy, .. }) = cli.command else {
                panic!("expected compile command");
            };
            assert_eq!(policy, expected);
        }
    }

    #[test]
    fn compile_link_explanation_outputs_parse_together() {
        let cli = Cli::try_parse_from([
            "runmat",
            "compile",
            "main.m",
            "--explain-link",
            "--link-plan-json",
            "build/main.link.json",
        ])
        .unwrap();
        let Some(Commands::Compile {
            explain_link,
            link_plan_json,
            ..
        }) = cli.command
        else {
            panic!("expected compile command");
        };
        assert!(explain_link);
        assert_eq!(
            link_plan_json.as_deref(),
            Some(std::path::Path::new("build/main.link.json"))
        );
    }

    #[test]
    fn selected_compile_requires_separate_pack_and_host_admission_inputs() {
        let selected = Cli::try_parse_from([
            "runmat",
            "compile",
            "main.m",
            "--policy",
            "dynamic-runtime",
            "--selected-pack-manifest",
            "pack.json",
            "--selected-components",
            "pack-bytes",
            "--selected-host-admission",
            "qualified-host.json",
        ])
        .unwrap();
        assert!(matches!(
            selected.command,
            Some(Commands::Compile {
                selected_pack_manifest: Some(_),
                selected_components: Some(_),
                selected_host_admission: Some(_),
                linker: None,
                ..
            })
        ));
        assert!(Cli::try_parse_from([
            "runmat",
            "compile",
            "main.m",
            "--selected-pack-manifest",
            "pack.json",
        ])
        .is_err());
        assert!(Cli::try_parse_from([
            "runmat",
            "compile",
            "main.m",
            "--selected-pack-manifest",
            "pack.json",
            "--selected-components",
            "pack-bytes",
            "--selected-host-admission",
            "host.json",
            "--linker",
            "/usr/bin/cc",
        ])
        .is_err());
    }

    #[test]
    fn durable_batch_commands_parse_without_entering_script_mode() {
        let cli = Cli::try_parse_from([
            "runmat",
            "batch",
            "submit",
            "job.m",
            "--idempotency-key",
            "stable",
            "--retention-hours",
            "24",
            "--json",
            "--",
            "input",
        ])
        .unwrap();
        let Some(Commands::Batch {
            batch_command:
                BatchCommand::Submit {
                    file,
                    idempotency_key,
                    retention_hours,
                    json,
                    args,
                },
        }) = cli.command
        else {
            panic!("expected durable batch submit");
        };
        assert_eq!(file, std::path::Path::new("job.m"));
        assert_eq!(idempotency_key.as_deref(), Some("stable"));
        assert_eq!(retention_hours, 24);
        assert!(json);
        assert_eq!(args, ["input"]);
    }

    #[test]
    fn cluster_management_commands_parse_without_entering_script_mode() {
        let cli = Cli::try_parse_from([
            "runmat",
            "cluster",
            "state",
            "--org",
            "01234567-89ab-cdef-0123-456789abcdef",
            "clu_0123456789abcdef0123456789abcdef",
            "draining",
            "--json",
        ])
        .unwrap();
        let Some(Commands::Cluster {
            cluster_command:
                ClusterCommand::State {
                    cluster,
                    state,
                    json,
                    ..
                },
        }) = cli.command
        else {
            panic!("expected cluster state command");
        };
        assert_eq!(cluster, "clu_0123456789abcdef0123456789abcdef");
        assert!(matches!(state, ClusterStateArg::Draining));
        assert!(json);
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ConfigFormat {
    Toml,
    Json,
}
