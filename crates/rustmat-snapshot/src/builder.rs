//! Snapshot builder for creating optimized snapshots of the standard library
//!
//! High-performance builder that preloads, analyzes, and optimizes all standard
//! library components into a single snapshot file.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::RwLock;

use crate::compression::{CompressionConfig, CompressionEngine};
use crate::format::*;
use crate::validation::SnapshotValidator;
use crate::*;

/// Snapshot builder with progressive enhancement
pub struct SnapshotBuilder {
    /// Configuration
    config: SnapshotConfig,

    /// Compression engine
    compression: CompressionEngine,

    /// Validation engine
    #[cfg(feature = "validation")]
    validator: SnapshotValidator,

    /// Build statistics
    stats: Arc<RwLock<BuildStats>>,

    /// Progress reporting
    progress: Option<ProgressBar>,
}

/// Build statistics
#[derive(Debug, Default)]
pub struct BuildStats {
    /// Start time
    pub start_time: Option<Instant>,

    /// Phase timings
    pub phase_times: HashMap<String, Duration>,

    /// Memory usage tracking
    pub memory_usage: Vec<(String, usize)>,

    /// Items processed
    pub items_processed: HashMap<String, usize>,

    /// Errors encountered
    pub errors: Vec<String>,

    /// Warnings
    pub warnings: Vec<String>,
}

/// Build phases for progress tracking
#[derive(Debug, Clone)]
pub enum BuildPhase {
    Initialization,
    BuiltinRegistration,
    HirCaching,
    BytecodeCaching,
    GcPresetCaching,
    OptimizationAnalysis,
    Compression,
    Validation,
    Serialization,
    Finalization,
}

impl BuildPhase {
    fn name(&self) -> &'static str {
        match self {
            BuildPhase::Initialization => "Initialization",
            BuildPhase::BuiltinRegistration => "Builtin Registration",
            BuildPhase::HirCaching => "HIR Caching",
            BuildPhase::BytecodeCaching => "Bytecode Caching",
            BuildPhase::GcPresetCaching => "GC Preset Caching",
            BuildPhase::OptimizationAnalysis => "Optimization Analysis",
            BuildPhase::Compression => "Compression",
            BuildPhase::Validation => "Validation",
            BuildPhase::Serialization => "Serialization",
            BuildPhase::Finalization => "Finalization",
        }
    }

    fn weight(&self) -> u64 {
        match self {
            BuildPhase::Initialization => 5,
            BuildPhase::BuiltinRegistration => 15,
            BuildPhase::HirCaching => 20,
            BuildPhase::BytecodeCaching => 25,
            BuildPhase::GcPresetCaching => 5,
            BuildPhase::OptimizationAnalysis => 10,
            BuildPhase::Compression => 10,
            BuildPhase::Validation => 5,
            BuildPhase::Serialization => 3,
            BuildPhase::Finalization => 2,
        }
    }

    /// Check if this phase requires compression
    pub fn needs_compression(&self) -> bool {
        matches!(self, BuildPhase::Compression)
    }

    /// Check if this phase requires validation  
    pub fn needs_validation(&self) -> bool {
        matches!(self, BuildPhase::Validation)
    }

    /// Check if this phase involves serialization
    pub fn involves_serialization(&self) -> bool {
        matches!(self, BuildPhase::Serialization | BuildPhase::Finalization)
    }
}

impl SnapshotBuilder {
    /// Create a new snapshot builder
    pub fn new(config: SnapshotConfig) -> Self {
        let compression_config = CompressionConfig {
            default_level: config.compression_level,
            adaptive_selection: true,
            prefer_speed: false,
            ..CompressionConfig::default()
        };

        let compression = CompressionEngine::new(compression_config);

        #[cfg(feature = "validation")]
        let validator = SnapshotValidator::new();

        let progress = if config.progress_reporting {
            let pb = ProgressBar::new(100);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>3}/{len:3} {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        Self {
            config,
            compression,
            #[cfg(feature = "validation")]
            validator,
            stats: Arc::new(RwLock::new(BuildStats::default())),
            progress,
        }
    }

    /// Build and save snapshot to file
    pub fn build_and_save<P: AsRef<Path>>(&self, output_path: P) -> SnapshotResult<()> {
        let snapshot = self.build()?;
        self.save_snapshot(&snapshot, output_path)
    }

    /// Get compression engine for external use
    pub fn compression_engine(&self) -> &CompressionEngine {
        &self.compression
    }

    /// Get validator for external use
    #[cfg(feature = "validation")]
    pub fn validator(&self) -> &SnapshotValidator {
        &self.validator
    }

    /// Test all build phases for completeness
    #[cfg(test)]
    pub fn test_all_phases() -> Vec<BuildPhase> {
        vec![
            BuildPhase::Initialization,
            BuildPhase::BuiltinRegistration,
            BuildPhase::HirCaching,
            BuildPhase::BytecodeCaching,
            BuildPhase::GcPresetCaching,
            BuildPhase::OptimizationAnalysis,
            BuildPhase::Compression,
            BuildPhase::Validation,
            BuildPhase::Serialization,
            BuildPhase::Finalization,
        ]
    }

    /// Analyze build phase requirements
    pub fn analyze_phase_requirements(phase: &BuildPhase) -> String {
        let mut requirements = Vec::new();

        if phase.needs_compression() {
            requirements.push("compression engine");
        }
        if phase.needs_validation() {
            requirements.push("validation framework");
        }
        if phase.involves_serialization() {
            requirements.push("serialization support");
        }

        if requirements.is_empty() {
            "No special requirements".to_string()
        } else {
            format!("Requires: {}", requirements.join(", "))
        }
    }

    /// Build snapshot in memory
    pub fn build(&self) -> SnapshotResult<Snapshot> {
        self.start_build();

        let phases = [
            BuildPhase::Initialization,
            BuildPhase::BuiltinRegistration,
            BuildPhase::HirCaching,
            BuildPhase::BytecodeCaching,
            BuildPhase::GcPresetCaching,
            BuildPhase::OptimizationAnalysis,
            BuildPhase::Finalization,
        ];

        let mut current_progress = 0u64;
        let total_progress: u64 = phases.iter().map(|p| p.weight()).sum();

        // Initialize snapshot
        let mut snapshot = self.execute_phase(BuildPhase::Initialization, || {
            Ok(self.create_empty_snapshot())
        })?;
        current_progress += BuildPhase::Initialization.weight();
        self.update_progress(
            current_progress,
            total_progress,
            BuildPhase::Initialization.name(),
        );

        // Build builtin registry
        snapshot.builtins = self.execute_phase(BuildPhase::BuiltinRegistration, || {
            self.build_builtin_registry()
        })?;
        current_progress += BuildPhase::BuiltinRegistration.weight();
        self.update_progress(
            current_progress,
            total_progress,
            BuildPhase::BuiltinRegistration.name(),
        );

        // Build HIR cache
        snapshot.hir_cache =
            self.execute_phase(BuildPhase::HirCaching, || self.build_hir_cache())?;
        current_progress += BuildPhase::HirCaching.weight();
        self.update_progress(
            current_progress,
            total_progress,
            BuildPhase::HirCaching.name(),
        );

        // Build bytecode cache
        snapshot.bytecode_cache = self.execute_phase(BuildPhase::BytecodeCaching, || {
            self.build_bytecode_cache(&snapshot.hir_cache)
        })?;
        current_progress += BuildPhase::BytecodeCaching.weight();
        self.update_progress(
            current_progress,
            total_progress,
            BuildPhase::BytecodeCaching.name(),
        );

        // Build GC presets
        snapshot.gc_presets =
            self.execute_phase(BuildPhase::GcPresetCaching, || self.build_gc_presets())?;
        current_progress += BuildPhase::GcPresetCaching.weight();
        self.update_progress(
            current_progress,
            total_progress,
            BuildPhase::GcPresetCaching.name(),
        );

        // Generate optimization hints
        snapshot.optimization_hints = self
            .execute_phase(BuildPhase::OptimizationAnalysis, || {
                self.generate_optimization_hints(&snapshot)
            })?;
        current_progress += BuildPhase::OptimizationAnalysis.weight();
        self.update_progress(
            current_progress,
            total_progress,
            BuildPhase::OptimizationAnalysis.name(),
        );

        // Finalize snapshot
        self.execute_phase(BuildPhase::Finalization, || {
            self.finalize_snapshot(&mut snapshot)
        })?;
        current_progress += BuildPhase::Finalization.weight();
        self.update_progress(
            current_progress,
            total_progress,
            BuildPhase::Finalization.name(),
        );

        self.finish_build();

        Ok(snapshot)
    }

    /// Execute a build phase with timing and error handling
    fn execute_phase<T, F>(&self, phase: BuildPhase, f: F) -> SnapshotResult<T>
    where
        F: FnOnce() -> SnapshotResult<T>,
    {
        let start = Instant::now();
        log::info!("Starting build phase: {}", phase.name());

        let result = f().context(format!("Failed in phase: {}", phase.name()));

        let duration = start.elapsed();
        {
            let mut stats = self.stats.write();
            stats.phase_times.insert(phase.name().to_string(), duration);

            match &result {
                Ok(_) => {
                    log::info!("Completed build phase: {} in {:?}", phase.name(), duration);
                }
                Err(e) => {
                    let error_msg = format!("Failed in phase {}: {}", phase.name(), e);
                    log::error!("{error_msg}");
                    stats.errors.push(error_msg);
                }
            }
        }

        result.map_err(|e| SnapshotError::Configuration {
            message: e.to_string(),
        })
    }

    /// Create empty snapshot structure
    fn create_empty_snapshot(&self) -> Snapshot {
        Snapshot {
            metadata: SnapshotMetadata::current(),
            builtins: BuiltinRegistry {
                name_index: HashMap::new(),
                functions: Vec::new(),
                dispatch_table: Arc::new(RwLock::new(Vec::new())),
            },
            hir_cache: HirCache {
                functions: HashMap::new(),
                patterns: Vec::new(),
                type_cache: HashMap::new(),
            },
            bytecode_cache: BytecodeCache {
                stdlib_bytecode: HashMap::new(),
                operation_sequences: Vec::new(),
                hotspots: Vec::new(),
            },
            gc_presets: GcPresetCache {
                presets: HashMap::new(),
                default_preset: "default".to_string(),
                performance_profiles: HashMap::new(),
            },
            optimization_hints: OptimizationHints {
                jit_hints: Vec::new(),
                memory_hints: Vec::new(),
                execution_hints: Vec::new(),
            },
        }
    }

    /// Build optimized builtin function registry
    fn build_builtin_registry(&self) -> SnapshotResult<BuiltinRegistry> {
        log::info!("Building builtin function registry");

        let builtins = rustmat_builtins::builtins();
        let mut name_index = HashMap::new();
        let mut functions = Vec::new();
        let mut dispatch_table = Vec::new();

        for (index, builtin) in builtins.iter().enumerate() {
            name_index.insert(builtin.name.to_string(), index);

            // Analyze function characteristics
            let metadata = self.analyze_builtin_function(builtin)?;
            functions.push(metadata);

            dispatch_table.push(builtin.func);

            {
                let mut stats = self.stats.write();
                *stats
                    .items_processed
                    .entry("builtins".to_string())
                    .or_insert(0) += 1;
            }
        }

        log::info!("Registered {} builtin functions", functions.len());

        Ok(BuiltinRegistry {
            name_index,
            functions,
            dispatch_table: Arc::new(RwLock::new(dispatch_table)),
        })
    }

    /// Analyze builtin function characteristics
    fn analyze_builtin_function(
        &self,
        builtin: &rustmat_builtins::Builtin,
    ) -> SnapshotResult<BuiltinMetadata> {
        // Infer characteristics from function name
        let category = self.infer_builtin_category(builtin.name);
        let complexity = self.infer_computational_complexity(builtin.name);
        let optimization_level = self.infer_optimization_level(builtin.name, &category);

        // For now, assume most functions take 1-2 arguments
        // In a real implementation, this would use reflection or metadata
        let arity = if builtin.name.ends_with("mul") || builtin.name.contains("dot") {
            BuiltinArity::Exact(2)
        } else if builtin.name == "norm"
            || builtin.name.starts_with("sin")
            || builtin.name.starts_with("cos")
        {
            BuiltinArity::Exact(1)
        } else {
            BuiltinArity::Range(1, 3)
        };

        Ok(BuiltinMetadata {
            name: builtin.name.to_string(),
            arity,
            category,
            complexity,
            optimization_level,
        })
    }

    /// Infer builtin function category
    fn infer_builtin_category(&self, name: &str) -> BuiltinCategory {
        if name.contains("sin")
            || name.contains("cos")
            || name.contains("tan")
            || name.contains("atan")
            || name.contains("asin")
            || name.contains("acos")
        {
            BuiltinCategory::Trigonometric
        } else if name.contains("mat")
            || name.contains("dot")
            || name.contains("norm")
            || name.contains("inv")
            || name.contains("det")
        {
            BuiltinCategory::LinearAlgebra
        } else if name.contains("mean") || name.contains("std") || name.contains("var") {
            BuiltinCategory::Statistics
        } else if name.contains("transpose") || name.contains("reshape") || name.contains("size") {
            BuiltinCategory::MatrixOps
        } else if name == "max"
            || name == "min"
            || name.contains("equal")
            || name.contains("greater")
            || name.contains("less")
        {
            BuiltinCategory::Comparison
        } else if name.contains("sqrt")
            || name.contains("exp")
            || name.contains("log")
            || name.contains("abs")
            || name.contains("pow")
        {
            BuiltinCategory::Math
        } else {
            BuiltinCategory::Utility
        }
    }

    /// Infer computational complexity
    fn infer_computational_complexity(&self, name: &str) -> ComputationalComplexity {
        if name.contains("matmul") || name.contains("inv") || name.contains("det") {
            ComputationalComplexity::Cubic
        } else if name.contains("mat") && !name.contains("matmul") {
            ComputationalComplexity::Quadratic
        } else if name.contains("dot") || name.contains("norm") || name.contains("sum") {
            ComputationalComplexity::Linear
        } else {
            ComputationalComplexity::Constant
        }
    }

    /// Infer optimization level
    fn infer_optimization_level(
        &self,
        _name: &str,
        category: &BuiltinCategory,
    ) -> OptimizationLevel {
        match category {
            BuiltinCategory::LinearAlgebra | BuiltinCategory::MatrixOps => {
                OptimizationLevel::MaxPerformance
            }
            BuiltinCategory::Math | BuiltinCategory::Trigonometric => OptimizationLevel::Aggressive,
            BuiltinCategory::Statistics => OptimizationLevel::Basic,
            _ => OptimizationLevel::None,
        }
    }

    /// Build HIR cache for standard library functions
    fn build_hir_cache(&self) -> SnapshotResult<HirCache> {
        log::info!("Building HIR cache");

        let mut functions = HashMap::new();
        let mut patterns = Vec::new();
        let mut type_cache = HashMap::new();

        // Cache common standard library functions
        let stdlib_functions = self.get_stdlib_function_sources();

        for (name, source) in stdlib_functions {
            match self.compile_to_hir(&source) {
                Ok(hir) => {
                    // Extract type information
                    self.extract_type_info(&hir, &mut type_cache);

                    // Store HIR
                    functions.insert(name.clone(), hir);

                    {
                        let mut stats = self.stats.write();
                        *stats
                            .items_processed
                            .entry("hir_functions".to_string())
                            .or_insert(0) += 1;
                    }
                }
                Err(e) => {
                    let warning = format!("Failed to compile {name} to HIR: {e}");
                    log::warn!("{warning}");

                    let mut stats = self.stats.write();
                    stats.warnings.push(warning);
                }
            }
        }

        // Generate common patterns
        patterns.extend(self.generate_common_patterns());

        log::info!(
            "Cached {} HIR functions and {} patterns",
            functions.len(),
            patterns.len()
        );

        Ok(HirCache {
            functions,
            patterns,
            type_cache,
        })
    }

    /// Get standard library function sources
    fn get_stdlib_function_sources(&self) -> Vec<(String, String)> {
        vec![
            (
                "zeros".to_string(),
                "function z = zeros(m, n); z = zeros(m, n); end".to_string(),
            ),
            (
                "ones".to_string(),
                "function o = ones(m, n); o = ones(m, n); end".to_string(),
            ),
            (
                "eye".to_string(),
                "function i = eye(n); i = eye(n); end".to_string(),
            ),
            (
                "sum_vec".to_string(),
                "function s = sum_vec(v); s = 0; for i = 1:length(v); s = s + v(i); end; end"
                    .to_string(),
            ),
            (
                "mean_vec".to_string(),
                "function m = mean_vec(v); m = sum_vec(v) / length(v); end".to_string(),
            ),
        ]
    }

    /// Compile source to HIR
    fn compile_to_hir(&self, source: &str) -> Result<rustmat_hir::HirProgram> {
        let ast = rustmat_parser::parse(source).map_err(|e| anyhow::anyhow!(e))?;
        let hir = rustmat_hir::lower(&ast).map_err(|e| anyhow::anyhow!(e))?;
        Ok(hir)
    }

    /// Extract type information from HIR
    fn extract_type_info(
        &self,
        _hir: &rustmat_hir::HirProgram,
        _type_cache: &mut HashMap<String, rustmat_hir::Type>,
    ) {
        // Type extraction would analyze HIR and populate type cache
        // For now, this is a placeholder
    }

    /// Generate common HIR patterns
    fn generate_common_patterns(&self) -> Vec<HirPattern> {
        vec![
            // Common loop patterns
            HirPattern {
                name: "simple_for_loop".to_string(),
                pattern: self.create_pattern_hir("for i = 1:n; x = x + 1; end"),
                frequency: 1000,
                optimization_priority: OptimizationLevel::Aggressive,
            },
            // Common matrix operations
            HirPattern {
                name: "matrix_multiply".to_string(),
                pattern: self.create_pattern_hir("C = A * B"),
                frequency: 500,
                optimization_priority: OptimizationLevel::MaxPerformance,
            },
        ]
    }

    /// Create HIR pattern (simplified)
    fn create_pattern_hir(&self, source: &str) -> rustmat_hir::HirProgram {
        self.compile_to_hir(source).unwrap_or_else(|_| {
            // Fallback to empty program
            rustmat_hir::HirProgram { body: Vec::new() }
        })
    }

    /// Build bytecode cache
    fn build_bytecode_cache(&self, hir_cache: &HirCache) -> SnapshotResult<BytecodeCache> {
        log::info!("Building bytecode cache");

        let mut stdlib_bytecode = HashMap::new();
        let mut operation_sequences = Vec::new();
        let mut hotspots = Vec::new();

        // Compile HIR functions to bytecode
        for (name, hir) in &hir_cache.functions {
            match rustmat_ignition::compile(hir) {
                Ok(bytecode) => {
                    stdlib_bytecode.insert(name.clone(), bytecode);

                    {
                        let mut stats = self.stats.write();
                        *stats
                            .items_processed
                            .entry("bytecode_functions".to_string())
                            .or_insert(0) += 1;
                    }
                }
                Err(e) => {
                    let warning = format!("Failed to compile {name} to bytecode: {e}");
                    log::warn!("{warning}");

                    let mut stats = self.stats.write();
                    stats.warnings.push(warning);
                }
            }
        }

        // Generate common operation sequences
        operation_sequences.extend(self.generate_operation_sequences());

        // Identify potential hotspots
        hotspots.extend(self.identify_hotspot_bytecode(&stdlib_bytecode));

        log::info!(
            "Cached {} bytecode functions, {} sequences, {} hotspots",
            stdlib_bytecode.len(),
            operation_sequences.len(),
            hotspots.len()
        );

        Ok(BytecodeCache {
            stdlib_bytecode,
            operation_sequences,
            hotspots,
        })
    }

    /// Generate common operation sequences
    fn generate_operation_sequences(&self) -> Vec<BytecodeSequence> {
        vec![
            BytecodeSequence {
                name: "scalar_add".to_string(),
                bytecode: self.create_sequence_bytecode("x = a + b"),
                usage_count: 10000,
                average_execution_time: Duration::from_nanos(100),
            },
            BytecodeSequence {
                name: "scalar_multiply".to_string(),
                bytecode: self.create_sequence_bytecode("x = a * b"),
                usage_count: 8000,
                average_execution_time: Duration::from_nanos(120),
            },
        ]
    }

    /// Create bytecode for sequence
    fn create_sequence_bytecode(&self, source: &str) -> rustmat_ignition::Bytecode {
        match self.compile_to_hir(source) {
            Ok(hir) => {
                rustmat_ignition::compile(&hir).unwrap_or_else(|_| rustmat_ignition::Bytecode {
                    instructions: Vec::new(),
                    var_count: 0,
                })
            }
            Err(_) => rustmat_ignition::Bytecode {
                instructions: Vec::new(),
                var_count: 0,
            },
        }
    }

    /// Identify hotspot bytecode for JIT optimization
    fn identify_hotspot_bytecode(
        &self,
        stdlib_bytecode: &HashMap<String, rustmat_ignition::Bytecode>,
    ) -> Vec<HotspotBytecode> {
        let mut hotspots = Vec::new();

        for (name, bytecode) in stdlib_bytecode {
            if self.is_hotspot_candidate(name, bytecode) {
                hotspots.push(HotspotBytecode {
                    name: name.clone(),
                    bytecode: bytecode.clone(),
                    execution_frequency: self.estimate_execution_frequency(name),
                    jit_compilation_threshold: self.determine_jit_threshold(name),
                    optimization_hints: self.generate_bytecode_optimization_hints(name, bytecode),
                });
            }
        }

        hotspots
    }

    /// Check if bytecode is a hotspot candidate
    fn is_hotspot_candidate(&self, name: &str, bytecode: &rustmat_ignition::Bytecode) -> bool {
        // Functions with loops or many instructions are good candidates
        bytecode.instructions.len() > 10
            || name.contains("loop")
            || name.contains("mat")
            || bytecode.instructions.iter().any(|instr| {
                matches!(
                    instr,
                    rustmat_ignition::Instr::Jump(_) | rustmat_ignition::Instr::JumpIfFalse(_)
                )
            })
    }

    /// Estimate execution frequency for function
    fn estimate_execution_frequency(&self, name: &str) -> u64 {
        // Heuristic based on function type
        if name.contains("mat") || name.contains("linear") {
            1000 // High frequency for matrix operations
        } else if name.contains("loop") {
            500 // Medium frequency for loops
        } else {
            100 // Low frequency for utilities
        }
    }

    /// Determine JIT compilation threshold
    fn determine_jit_threshold(&self, name: &str) -> u32 {
        if name.contains("mat") {
            5 // Compile matrix operations quickly
        } else if name.contains("loop") {
            10 // Medium threshold for loops
        } else {
            20 // Higher threshold for simple functions
        }
    }

    /// Generate optimization hints for bytecode
    fn generate_bytecode_optimization_hints(
        &self,
        name: &str,
        _bytecode: &rustmat_ignition::Bytecode,
    ) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        if name.contains("mat") {
            hints.push(OptimizationHint {
                hint_type: "vectorization".to_string(),
                parameters: [("target".to_string(), "simd".to_string())]
                    .iter()
                    .cloned()
                    .collect(),
                expected_speedup: 4.0,
            });
        }

        if name.contains("loop") {
            hints.push(OptimizationHint {
                hint_type: "loop_unrolling".to_string(),
                parameters: [("factor".to_string(), "4".to_string())]
                    .iter()
                    .cloned()
                    .collect(),
                expected_speedup: 2.0,
            });
        }

        hints
    }

    /// Build GC preset cache
    fn build_gc_presets(&self) -> SnapshotResult<GcPresetCache> {
        log::info!("Building GC preset cache");

        let mut presets = HashMap::new();
        let mut performance_profiles = HashMap::new();

        // Create standard presets
        presets.insert("default".to_string(), rustmat_gc::GcConfig::default());
        presets.insert(
            "low-latency".to_string(),
            rustmat_gc::GcConfig::low_latency(),
        );
        presets.insert(
            "high-throughput".to_string(),
            rustmat_gc::GcConfig::high_throughput(),
        );
        presets.insert("low-memory".to_string(), rustmat_gc::GcConfig::low_memory());
        presets.insert("debug".to_string(), rustmat_gc::GcConfig::debug());

        // Create performance profiles
        for preset_name in presets.keys() {
            performance_profiles.insert(
                preset_name.clone(),
                self.create_gc_performance_profile(preset_name),
            );
        }

        log::info!("Created {} GC presets", presets.len());

        Ok(GcPresetCache {
            presets,
            default_preset: "default".to_string(),
            performance_profiles,
        })
    }

    /// Create performance profile for GC preset
    fn create_gc_performance_profile(&self, preset_name: &str) -> GcPerformanceProfile {
        // Estimated performance characteristics
        match preset_name {
            "low-latency" => GcPerformanceProfile {
                average_allocation_rate: 1000000.0, // allocations/sec
                average_collection_time: Duration::from_micros(100),
                memory_overhead: 0.1,
                throughput_impact: 0.05,
            },
            "high-throughput" => GcPerformanceProfile {
                average_allocation_rate: 2000000.0,
                average_collection_time: Duration::from_millis(10),
                memory_overhead: 0.2,
                throughput_impact: 0.02,
            },
            "low-memory" => GcPerformanceProfile {
                average_allocation_rate: 500000.0,
                average_collection_time: Duration::from_millis(5),
                memory_overhead: 0.05,
                throughput_impact: 0.1,
            },
            _ => GcPerformanceProfile {
                average_allocation_rate: 800000.0,
                average_collection_time: Duration::from_millis(2),
                memory_overhead: 0.15,
                throughput_impact: 0.08,
            },
        }
    }

    /// Generate optimization hints
    fn generate_optimization_hints(
        &self,
        snapshot: &Snapshot,
    ) -> SnapshotResult<OptimizationHints> {
        log::info!("Generating optimization hints");

        let mut jit_hints = Vec::new();
        let mut memory_hints = Vec::new();
        let mut execution_hints = Vec::new();

        // Generate JIT hints based on builtins
        for builtin in &snapshot.builtins.functions {
            if matches!(
                builtin.optimization_level,
                OptimizationLevel::Aggressive | OptimizationLevel::MaxPerformance
            ) {
                jit_hints.push(JitHint {
                    pattern: builtin.name.clone(),
                    hint_type: self.determine_jit_hint_type(&builtin.category),
                    priority: builtin.optimization_level.clone(),
                    expected_performance_gain: self
                        .estimate_jit_performance_gain(&builtin.complexity),
                });
            }
        }

        // Generate memory hints
        memory_hints.extend(self.generate_memory_hints());

        // Generate execution hints
        execution_hints.extend(self.generate_execution_hints(&snapshot.bytecode_cache));

        log::info!(
            "Generated {} JIT hints, {} memory hints, {} execution hints",
            jit_hints.len(),
            memory_hints.len(),
            execution_hints.len()
        );

        Ok(OptimizationHints {
            jit_hints,
            memory_hints,
            execution_hints,
        })
    }

    /// Determine JIT hint type for builtin category
    fn determine_jit_hint_type(&self, category: &BuiltinCategory) -> JitHintType {
        match category {
            BuiltinCategory::LinearAlgebra | BuiltinCategory::MatrixOps => {
                JitHintType::VectorizeCandidate
            }
            BuiltinCategory::Math | BuiltinCategory::Trigonometric => JitHintType::InlineCandidate,
            _ => JitHintType::ConstantFolding,
        }
    }

    /// Estimate JIT performance gain
    fn estimate_jit_performance_gain(&self, complexity: &ComputationalComplexity) -> f64 {
        match complexity {
            ComputationalComplexity::Constant => 1.5,
            ComputationalComplexity::Linear => 3.0,
            ComputationalComplexity::Quadratic => 5.0,
            ComputationalComplexity::Cubic => 8.0,
            ComputationalComplexity::Exponential => 10.0,
        }
    }

    /// Generate memory optimization hints
    fn generate_memory_hints(&self) -> Vec<MemoryHint> {
        vec![
            MemoryHint {
                data_structure: "matrix_data".to_string(),
                hint_type: MemoryHintType::AlignmentOptimization,
                alignment: 64, // Cache line alignment
                prefetch_pattern: PrefetchPattern::Sequential,
            },
            MemoryHint {
                data_structure: "builtin_dispatch".to_string(),
                hint_type: MemoryHintType::CacheLocalityOptimization,
                alignment: 8,
                prefetch_pattern: PrefetchPattern::Random,
            },
        ]
    }

    /// Generate execution hints
    fn generate_execution_hints(&self, bytecode_cache: &BytecodeCache) -> Vec<ExecutionHint> {
        let mut hints = Vec::new();

        for hotspot in &bytecode_cache.hotspots {
            hints.push(ExecutionHint {
                pattern: hotspot.name.clone(),
                hint_type: ExecutionHintType::HotPath,
                frequency: hotspot.execution_frequency,
                optimization_potential: hotspot
                    .optimization_hints
                    .iter()
                    .map(|h| h.expected_speedup)
                    .fold(0.0, f64::max),
            });
        }

        hints
    }

    /// Finalize snapshot with metadata
    fn finalize_snapshot(&self, snapshot: &mut Snapshot) -> SnapshotResult<()> {
        log::info!("Finalizing snapshot");

        // Update performance metrics
        let stats = self.stats.read();
        snapshot.metadata.performance_metrics = PerformanceMetrics {
            creation_time: stats
                .start_time
                .map_or(Duration::ZERO, |start| start.elapsed()),
            builtin_count: snapshot.builtins.functions.len(),
            hir_cache_entries: snapshot.hir_cache.functions.len(),
            bytecode_cache_entries: snapshot.bytecode_cache.stdlib_bytecode.len(),
            uncompressed_size: bincode::serialized_size(snapshot).unwrap_or(0) as usize,
            compression_ratio: 1.0, // Will be updated after compression
            peak_memory_usage: self.estimate_peak_memory_usage(),
        };

        Ok(())
    }

    /// Save snapshot to file
    fn save_snapshot<P: AsRef<Path>>(
        &self,
        snapshot: &Snapshot,
        output_path: P,
    ) -> SnapshotResult<()> {
        log::info!("Saving snapshot to {}", output_path.as_ref().display());

        // Serialize snapshot
        let serialized = bincode::serialize(snapshot).map_err(SnapshotError::Serialization)?;

        // Store original serialized size before compression
        let uncompressed_size = serialized.len();

        // Compress if enabled
        let (data, _compression_info) = if self.config.compression_enabled {
            let mut compression = CompressionEngine::new(CompressionConfig {
                default_level: self.config.compression_level,
                ..CompressionConfig::default()
            });

            let result = compression.compress(&serialized)?;
            (result.data, Some(result.info))
        } else {
            (serialized, None)
        };

        // Create format
        let mut header = SnapshotHeader::new(snapshot.metadata.clone());

        // Update data info with actual sizes
        header.data_info.compressed_size = data.len();
        header.data_info.uncompressed_size = uncompressed_size;
        if self.config.compression_enabled {
            header.data_info.compression = _compression_info.unwrap_or_else(|| CompressionInfo {
                algorithm: format::CompressionAlgorithm::None,
                level: 0,
                parameters: std::collections::HashMap::new(),
            });
        } else {
            header.data_info.compression = CompressionInfo {
                algorithm: format::CompressionAlgorithm::None,
                level: 0,
                parameters: std::collections::HashMap::new(),
            };
        }

        let mut format = SnapshotFormat::new(header, data);

        // Add checksum if validation enabled
        #[cfg(feature = "validation")]
        if self.config.validation_enabled {
            format = format.with_checksum(crate::format::ChecksumAlgorithm::Sha256)?;
        }

        // Write to file
        self.write_snapshot_file(&format, output_path)?;

        log::info!("Snapshot saved successfully");
        Ok(())
    }

    /// Write snapshot format to file
    fn write_snapshot_file<P: AsRef<Path>>(
        &self,
        format: &SnapshotFormat,
        output_path: P,
    ) -> SnapshotResult<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(output_path)?;

        // Serialize header
        let header_data = bincode::serialize(&format.header)?;

        // Write header size first (4 bytes, little-endian)
        let header_size = header_data.len() as u32;
        file.write_all(&header_size.to_le_bytes())?;

        // Write header
        file.write_all(&header_data)?;

        // Write data
        file.write_all(&format.data)?;

        // Write checksum if present
        if let Some(checksum) = &format.checksum {
            file.write_all(checksum)?;
        }

        file.sync_all()?;
        Ok(())
    }

    /// Start build process
    fn start_build(&self) {
        {
            let mut stats = self.stats.write();
            stats.start_time = Some(Instant::now());
        }

        log::info!("Starting snapshot build");

        if let Some(ref progress) = self.progress {
            progress.set_message("Initializing...");
        }
    }

    /// Finish build process
    fn finish_build(&self) {
        if let Some(ref progress) = self.progress {
            progress.finish_with_message("Snapshot build completed!");
        }

        let stats = self.stats.read();
        if let Some(start_time) = stats.start_time {
            let total_time = start_time.elapsed();
            log::info!("Snapshot build completed in {total_time:?}");
        }
    }

    /// Update progress
    fn update_progress(&self, current: u64, total: u64, message: &str) {
        if let Some(ref progress) = self.progress {
            progress.set_position((current * 100) / total);
            progress.set_message(message.to_string());
        }
    }

    /// Estimate peak memory usage
    fn estimate_peak_memory_usage(&self) -> usize {
        // Simplified estimation
        std::mem::size_of::<Snapshot>() * 2 // Rough estimate
    }

    /// Get build statistics
    pub fn stats(&self) -> BuildStats {
        // Clone the data inside the lock
        let stats = self.stats.read();
        BuildStats {
            start_time: stats.start_time,
            phase_times: stats.phase_times.clone(),
            memory_usage: stats.memory_usage.clone(),
            items_processed: stats.items_processed.clone(),
            errors: stats.errors.clone(),
            warnings: stats.warnings.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_builder_creation() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        let stats = builder.stats();
        assert!(stats.start_time.is_none());
        assert!(stats.errors.is_empty());
    }

    #[test]
    fn test_builtin_analysis() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        let builtin = rustmat_builtins::Builtin {
            name: "matmul",
            func: |_| Ok(rustmat_builtins::Value::Num(0.0)),
        };

        let metadata = builder.analyze_builtin_function(&builtin).unwrap();
        assert_eq!(metadata.name, "matmul");
        assert!(matches!(metadata.category, BuiltinCategory::LinearAlgebra));
        assert!(matches!(
            metadata.complexity,
            ComputationalComplexity::Cubic
        ));
    }

    #[test]
    fn test_category_inference() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        assert!(matches!(
            builder.infer_builtin_category("sin"),
            BuiltinCategory::Trigonometric
        ));
        assert!(matches!(
            builder.infer_builtin_category("matmul"),
            BuiltinCategory::LinearAlgebra
        ));
        assert!(matches!(
            builder.infer_builtin_category("max"),
            BuiltinCategory::Comparison
        ));
    }

    #[test]
    fn test_complexity_inference() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        assert!(matches!(
            builder.infer_computational_complexity("matmul"),
            ComputationalComplexity::Cubic
        ));
        assert!(matches!(
            builder.infer_computational_complexity("dot"),
            ComputationalComplexity::Linear
        ));
        assert!(matches!(
            builder.infer_computational_complexity("abs"),
            ComputationalComplexity::Constant
        ));
    }

    #[test]
    fn test_empty_snapshot_creation() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        let snapshot = builder.create_empty_snapshot();
        assert!(snapshot.builtins.functions.is_empty());
        assert!(snapshot.hir_cache.functions.is_empty());
        assert!(snapshot.bytecode_cache.stdlib_bytecode.is_empty());
    }

    #[test]
    fn test_gc_performance_profile() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        let profile = builder.create_gc_performance_profile("low-latency");
        assert!(profile.average_collection_time < Duration::from_millis(1));
        assert!(profile.memory_overhead < 0.2);
    }

    #[test]
    fn test_build_phases() {
        // Test all build phases are properly constructed
        let phases = SnapshotBuilder::test_all_phases();
        assert_eq!(phases.len(), 10);

        // Test phase analysis
        for phase in &phases {
            let requirements = SnapshotBuilder::analyze_phase_requirements(phase);
            assert!(!requirements.is_empty());

            // Test specific phase methods
            match phase {
                BuildPhase::Compression => assert!(phase.needs_compression()),
                BuildPhase::Validation => assert!(phase.needs_validation()),
                BuildPhase::Serialization => assert!(phase.involves_serialization()),
                BuildPhase::Finalization => assert!(phase.involves_serialization()),
                _ => {
                    assert!(!phase.needs_compression());
                    assert!(!phase.needs_validation());
                }
            }
        }
    }

    #[test]
    fn test_compression_engine_access() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        // Test that we can access the compression engine
        let _engine = builder.compression_engine();
    }

    #[cfg(feature = "validation")]
    #[test]
    fn test_validator_access() {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);

        // Test that we can access the validator
        let _validator = builder.validator();
    }
}
