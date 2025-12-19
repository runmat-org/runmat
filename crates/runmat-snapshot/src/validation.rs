//! Snapshot validation and integrity checking
//!
//! Provides comprehensive validation for snapshot files including
//! format validation, integrity checks, and compatibility verification.

use runmat_time::Instant;
use std::collections::HashMap;
use std::time::Duration;

use crate::format::*;
use crate::{Snapshot, SnapshotResult};

/// Snapshot validator with comprehensive checks
#[derive(Default)]
pub struct SnapshotValidator {
    /// Validation configuration
    config: ValidationConfig,

    /// Validation statistics
    stats: ValidationStats,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable format validation
    pub format_validation: bool,

    /// Enable integrity checking
    pub integrity_checking: bool,

    /// Enable compatibility checking
    pub compatibility_checking: bool,

    /// Enable performance validation
    pub performance_validation: bool,

    /// Maximum validation time
    pub max_validation_time: Duration,

    /// Strict mode (fail on warnings)
    pub strict_mode: bool,
}

/// Validation statistics
#[derive(Debug, Default)]
pub struct ValidationStats {
    /// Checks performed
    pub checks_performed: HashMap<String, u64>,

    /// Validation time by check type
    pub check_times: HashMap<String, Duration>,

    /// Total validation time
    pub total_time: Duration,

    /// Errors found
    pub errors: Vec<ValidationError>,

    /// Warnings found
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub message: String,
    pub location: Option<String>,
    pub severity: ErrorSeverity,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub warning_type: ValidationWarningType,
    pub message: String,
    pub recommendation: Option<String>,
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    FormatError,
    IntegrityError,
    CompatibilityError,
    PerformanceError,
    ConfigurationError,
}

/// Validation warning types
#[derive(Debug, Clone)]
pub enum ValidationWarningType {
    PerformanceWarning,
    CompatibilityWarning,
    ConfigurationWarning,
    DeprecationWarning,
}

/// Error severity levels
#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Overall validation success
    pub is_valid: bool,

    /// Validation score (0-100)
    pub score: u8,

    /// Errors found
    pub errors: Vec<ValidationError>,

    /// Warnings found
    pub warnings: Vec<ValidationWarning>,

    /// Performance metrics
    pub metrics: ValidationMetrics,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Validation performance metrics
#[derive(Debug)]
pub struct ValidationMetrics {
    pub total_time: Duration,
    pub checks_performed: usize,
    pub throughput: f64, // checks per second
    pub memory_used: usize,
}

impl SnapshotValidator {
    /// Create a new snapshot validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            stats: ValidationStats::default(),
        }
    }

    /// Validate snapshot format
    pub fn validate_format(&mut self, format: &SnapshotFormat) -> SnapshotResult<ValidationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate header
        self.validate_header(&format.header, &mut errors, &mut warnings)?;

        // Validate data section
        self.validate_data_section(format, &mut errors, &mut warnings)?;

        // Validate checksum if present
        if format.header.checksum_info.is_some() {
            self.validate_checksum(format, &mut errors, &mut warnings)?;
        }

        let validation_time = start.elapsed();
        self.update_stats("format_validation", validation_time);

        Ok(self.create_validation_result(errors, warnings, validation_time, "format_validation"))
    }

    /// Validate snapshot content
    pub fn validate_content(&mut self, snapshot: &Snapshot) -> SnapshotResult<ValidationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate builtin registry
        self.validate_builtin_registry(&snapshot.builtins, &mut errors, &mut warnings)?;

        // Validate HIR cache
        self.validate_hir_cache(&snapshot.hir_cache, &mut errors, &mut warnings)?;

        // Validate bytecode cache
        self.validate_bytecode_cache(&snapshot.bytecode_cache, &mut errors, &mut warnings)?;

        // Validate GC presets
        self.validate_gc_presets(&snapshot.gc_presets, &mut errors, &mut warnings)?;

        // Validate optimization hints
        self.validate_optimization_hints(&snapshot.optimization_hints, &mut errors, &mut warnings)?;

        let validation_time = start.elapsed();
        self.update_stats("content_validation", validation_time);

        Ok(self.create_validation_result(errors, warnings, validation_time, "content_validation"))
    }

    /// Validate compatibility with current environment
    pub fn validate_compatibility(
        &mut self,
        snapshot: &Snapshot,
    ) -> SnapshotResult<ValidationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check version compatibility
        if !snapshot.metadata.is_compatible() {
            errors.push(ValidationError {
                error_type: ValidationErrorType::CompatibilityError,
                message: "Snapshot version is not compatible with current RunMat version"
                    .to_string(),
                location: Some("metadata.runmat_version".to_string()),
                severity: ErrorSeverity::High,
            });
        }

        // Check platform compatibility
        if !SnapshotHeader::new(snapshot.metadata.clone()).is_platform_compatible() {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::CompatibilityWarning,
                message: "Snapshot was created for a different platform".to_string(),
                recommendation: Some("Performance may be suboptimal".to_string()),
            });
        }

        // Check feature compatibility
        self.validate_feature_compatibility(&snapshot.metadata, &mut errors, &mut warnings)?;

        let validation_time = start.elapsed();
        self.update_stats("compatibility_validation", validation_time);

        Ok(self.create_validation_result(
            errors,
            warnings,
            validation_time,
            "compatibility_validation",
        ))
    }

    /// Validate header structure
    fn validate_header(
        &self,
        header: &SnapshotHeader,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut [ValidationWarning],
    ) -> SnapshotResult<()> {
        // Validate magic number
        if header.magic != *SNAPSHOT_MAGIC {
            errors.push(ValidationError {
                error_type: ValidationErrorType::FormatError,
                message: "Invalid magic number in header".to_string(),
                location: Some("header.magic".to_string()),
                severity: ErrorSeverity::Critical,
            });
        }

        // Validate version
        if header.version > SNAPSHOT_VERSION {
            errors.push(ValidationError {
                error_type: ValidationErrorType::FormatError,
                message: format!(
                    "Unsupported snapshot version: {} > {}",
                    header.version, SNAPSHOT_VERSION
                ),
                location: Some("header.version".to_string()),
                severity: ErrorSeverity::High,
            });
        }

        // Validate data section info
        if header.data_info.uncompressed_size == 0 {
            errors.push(ValidationError {
                error_type: ValidationErrorType::FormatError,
                message: "Data section appears to be empty".to_string(),
                location: Some("header.data_info.uncompressed_size".to_string()),
                severity: ErrorSeverity::Medium,
            });
        }

        Ok(())
    }

    /// Validate data section
    fn validate_data_section(
        &self,
        format: &SnapshotFormat,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) -> SnapshotResult<()> {
        // Check data size consistency
        if (format.data.len() as u64) != format.header.data_info.compressed_size {
            errors.push(ValidationError {
                error_type: ValidationErrorType::FormatError,
                message: "Data size mismatch between header and actual data".to_string(),
                location: Some("data_section".to_string()),
                severity: ErrorSeverity::High,
            });
        }

        // Check compression ratio
        let compression_ratio =
            format.data.len() as f64 / format.header.data_info.uncompressed_size as f64;
        if compression_ratio > 1.0 {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceWarning,
                message: "Compression appears to have increased data size".to_string(),
                recommendation: Some("Consider disabling compression for this data".to_string()),
            });
        }

        Ok(())
    }

    /// Validate checksum
    fn validate_checksum(
        &self,
        format: &SnapshotFormat,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut [ValidationWarning],
    ) -> SnapshotResult<()> {
        match format.validate_checksum() {
            Ok(true) => {
                // Checksum is valid
            }
            Ok(false) => {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::IntegrityError,
                    message: "Checksum validation failed".to_string(),
                    location: Some("checksum".to_string()),
                    severity: ErrorSeverity::Critical,
                });
            }
            Err(e) => {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::IntegrityError,
                    message: format!("Checksum validation error: {e}"),
                    location: Some("checksum".to_string()),
                    severity: ErrorSeverity::High,
                });
            }
        }

        Ok(())
    }

    /// Validate builtin registry
    fn validate_builtin_registry(
        &self,
        registry: &crate::BuiltinRegistry,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) -> SnapshotResult<()> {
        // Validate that the name_index map points to matching entries within the functions list.
        for (name, &mapped_index) in &registry.name_index {
            match registry.functions.get(mapped_index) {
                Some(function) if function.name == *name => {}
                Some(_) => {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::FormatError,
                        message: format!(
                            "Builtin registry index mismatch: expected '{}' at position {}, found '{}'",
                            name, mapped_index, registry.functions[mapped_index].name
                        ),
                        location: Some(format!("builtins.functions[{mapped_index}]")),
                        severity: ErrorSeverity::Medium,
                    });
                }
                None => {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::FormatError,
                        message: format!(
                            "Builtin registry name_index points outside function list: '{}' -> {}",
                            name, mapped_index
                        ),
                        location: Some("builtins.name_index".to_string()),
                        severity: ErrorSeverity::Medium,
                    });
                }
            }
        }

        // Check for essential builtins
        let essential_builtins = ["abs", "sin", "cos", "sqrt", "max", "min"];
        for builtin in &essential_builtins {
            if !registry.name_index.contains_key(*builtin) {
                warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::ConfigurationWarning,
                    message: format!("Essential builtin '{builtin}' not found"),
                    recommendation: Some(
                        "Ensure all standard library components are included".to_string(),
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validate HIR cache
    fn validate_hir_cache(
        &self,
        cache: &crate::HirCache,
        _errors: &mut [ValidationError],
        warnings: &mut Vec<ValidationWarning>,
    ) -> SnapshotResult<()> {
        // Check cache effectiveness
        if cache.functions.is_empty() {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceWarning,
                message: "HIR cache is empty".to_string(),
                recommendation: Some(
                    "Consider caching common standard library functions".to_string(),
                ),
            });
        }

        // Check pattern effectiveness
        if cache.patterns.is_empty() {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceWarning,
                message: "No HIR patterns cached".to_string(),
                recommendation: Some("Consider caching common expression patterns".to_string()),
            });
        }

        Ok(())
    }

    /// Validate bytecode cache
    fn validate_bytecode_cache(
        &self,
        cache: &crate::BytecodeCache,
        _errors: &mut [ValidationError],
        warnings: &mut Vec<ValidationWarning>,
    ) -> SnapshotResult<()> {
        // Check cache content
        if cache.stdlib_bytecode.is_empty() {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceWarning,
                message: "Bytecode cache is empty".to_string(),
                recommendation: Some("Consider precompiling standard library bytecode".to_string()),
            });
        }

        // Check hotspot identification
        if cache.hotspots.is_empty() {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceWarning,
                message: "No hotspot bytecode identified".to_string(),
                recommendation: Some(
                    "Consider profiling to identify optimization candidates".to_string(),
                ),
            });
        }

        Ok(())
    }

    /// Validate GC presets
    fn validate_gc_presets(
        &self,
        presets: &crate::GcPresetCache,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) -> SnapshotResult<()> {
        // Check default preset exists
        if !presets.presets.contains_key(&presets.default_preset) {
            errors.push(ValidationError {
                error_type: ValidationErrorType::ConfigurationError,
                message: "Default GC preset not found".to_string(),
                location: Some("gc_presets.default_preset".to_string()),
                severity: ErrorSeverity::Medium,
            });
        }

        // Check performance profiles
        for preset_name in presets.presets.keys() {
            if !presets.performance_profiles.contains_key(preset_name) {
                warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::ConfigurationWarning,
                    message: format!("No performance profile for preset '{preset_name}'"),
                    recommendation: Some(
                        "Add performance characteristics for better optimization".to_string(),
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validate optimization hints
    fn validate_optimization_hints(
        &self,
        hints: &crate::OptimizationHints,
        _errors: &mut [ValidationError],
        warnings: &mut Vec<ValidationWarning>,
    ) -> SnapshotResult<()> {
        // Check hint completeness
        if hints.jit_hints.is_empty() {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceWarning,
                message: "No JIT optimization hints provided".to_string(),
                recommendation: Some(
                    "Consider analyzing code for JIT optimization opportunities".to_string(),
                ),
            });
        }

        if hints.memory_hints.is_empty() {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceWarning,
                message: "No memory optimization hints provided".to_string(),
                recommendation: Some("Consider memory layout optimizations".to_string()),
            });
        }

        Ok(())
    }

    /// Validate feature compatibility
    fn validate_feature_compatibility(
        &self,
        metadata: &SnapshotMetadata,
        _errors: &mut [ValidationError],
        warnings: &mut Vec<ValidationWarning>,
    ) -> SnapshotResult<()> {
        let current_features = SnapshotMetadata::current().feature_flags;

        // Check for missing features
        for feature in &metadata.feature_flags {
            if !current_features.contains(feature) {
                warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::CompatibilityWarning,
                    message: format!("Snapshot uses feature '{feature}' which is not available"),
                    recommendation: Some("Some functionality may be disabled".to_string()),
                });
            }
        }

        // Check for additional features
        for feature in &current_features {
            if !metadata.feature_flags.contains(feature) {
                warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::CompatibilityWarning,
                    message: format!("Current environment has feature '{feature}' not in snapshot"),
                    recommendation: Some(
                        "Consider rebuilding snapshot with current features".to_string(),
                    ),
                });
            }
        }

        Ok(())
    }

    /// Update validation statistics
    fn update_stats(&mut self, check_type: &str, duration: Duration) {
        *self
            .stats
            .checks_performed
            .entry(check_type.to_string())
            .or_insert(0) += 1;
        self.stats
            .check_times
            .insert(check_type.to_string(), duration);
        self.stats.total_time += duration;
    }

    /// Create validation result
    fn create_validation_result(
        &self,
        errors: Vec<ValidationError>,
        warnings: Vec<ValidationWarning>,
        validation_time: Duration,
        _check_type: &str,
    ) -> ValidationResult {
        let is_valid = errors.is_empty()
            || (!self.config.strict_mode
                && errors
                    .iter()
                    .all(|e| matches!(e.severity, ErrorSeverity::Low)));

        let score = self.calculate_validation_score(&errors, &warnings);

        let recommendations = self.generate_recommendations(&errors, &warnings);

        ValidationResult {
            is_valid,
            score,
            errors,
            warnings,
            metrics: ValidationMetrics {
                total_time: validation_time,
                checks_performed: 1,
                throughput: 1.0 / validation_time.as_secs_f64(),
                memory_used: std::mem::size_of::<Self>(),
            },
            recommendations,
        }
    }

    /// Calculate validation score
    fn calculate_validation_score(
        &self,
        errors: &[ValidationError],
        warnings: &[ValidationWarning],
    ) -> u8 {
        let mut score = 100u8;

        for error in errors {
            let penalty = match error.severity {
                ErrorSeverity::Critical => 50,
                ErrorSeverity::High => 20,
                ErrorSeverity::Medium => 10,
                ErrorSeverity::Low => 5,
            };
            score = score.saturating_sub(penalty);
        }

        // Warnings reduce score by 2 each
        score = score.saturating_sub((warnings.len() as u8) * 2);

        score
    }

    /// Generate recommendations based on errors and warnings
    fn generate_recommendations(
        &self,
        errors: &[ValidationError],
        warnings: &[ValidationWarning],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if errors
            .iter()
            .any(|e| matches!(e.error_type, ValidationErrorType::IntegrityError))
        {
            recommendations.push("Regenerate snapshot to fix integrity issues".to_string());
        }

        if errors
            .iter()
            .any(|e| matches!(e.error_type, ValidationErrorType::CompatibilityError))
        {
            recommendations.push("Update RunMat version or regenerate snapshot".to_string());
        }

        if warnings
            .iter()
            .any(|w| matches!(w.warning_type, ValidationWarningType::PerformanceWarning))
        {
            recommendations.push("Consider optimizing snapshot for better performance".to_string());
        }

        recommendations
    }

    /// Get validation statistics
    pub fn stats(&self) -> &ValidationStats {
        &self.stats
    }

    /// Reset validation statistics
    pub fn reset_stats(&mut self) {
        self.stats = ValidationStats::default();
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            format_validation: true,
            integrity_checking: true,
            compatibility_checking: true,
            performance_validation: true,
            max_validation_time: Duration::from_secs(30),
            strict_mode: false,
        }
    }
}

impl ValidationResult {
    /// Check if validation passed
    pub fn is_ok(&self) -> bool {
        self.is_valid
    }

    /// Get critical errors
    pub fn critical_errors(&self) -> Vec<&ValidationError> {
        self.errors
            .iter()
            .filter(|e| matches!(e.severity, ErrorSeverity::Critical))
            .collect()
    }

    /// Get performance warnings
    pub fn performance_warnings(&self) -> Vec<&ValidationWarning> {
        self.warnings
            .iter()
            .filter(|w| matches!(w.warning_type, ValidationWarningType::PerformanceWarning))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = SnapshotValidator::new();
        assert!(validator.config.format_validation);
        assert!(validator.config.integrity_checking);
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::default();
        assert!(config.format_validation);
        assert!(!config.strict_mode);
        assert!(config.max_validation_time > Duration::ZERO);
    }

    #[test]
    fn test_validation_score_calculation() {
        let validator = SnapshotValidator::new();

        // No errors or warnings = perfect score
        assert_eq!(validator.calculate_validation_score(&[], &[]), 100);

        // Critical error
        let critical_error = ValidationError {
            error_type: ValidationErrorType::IntegrityError,
            message: "Test error".to_string(),
            location: None,
            severity: ErrorSeverity::Critical,
        };
        assert_eq!(
            validator.calculate_validation_score(&[critical_error], &[]),
            50
        );

        // Warning
        let warning = ValidationWarning {
            warning_type: ValidationWarningType::PerformanceWarning,
            message: "Test warning".to_string(),
            recommendation: None,
        };
        assert_eq!(validator.calculate_validation_score(&[], &[warning]), 98);
    }

    #[test]
    fn test_header_validation() {
        let validator = SnapshotValidator::new();
        let metadata = SnapshotMetadata::current();
        let mut header = SnapshotHeader::new(metadata);

        // Set up proper data info to avoid validation errors
        header.data_info.uncompressed_size = 1024;
        header.data_info.compressed_size = 512;
        header.data_info.data_offset = 256;

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        validator
            .validate_header(&header, &mut errors, &mut warnings)
            .unwrap();
        assert!(errors.is_empty(), "Validation errors: {errors:?}");
        assert!(errors.is_empty());
    }

    #[test]
    fn test_invalid_magic_detection() {
        let validator = SnapshotValidator::new();
        let metadata = SnapshotMetadata::current();
        let mut header = SnapshotHeader::new(metadata);
        header.magic = [0; 8]; // Invalid magic

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        validator
            .validate_header(&header, &mut errors, &mut warnings)
            .unwrap();
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e.error_type, ValidationErrorType::FormatError)));
    }
}
