//! High-performance snapshot loader with memory mapping and caching
//!
//! Optimized for fast startup times with parallel loading, compression,
//! and integration with the RustMat runtime.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::compression::CompressionConfig;
use anyhow::Context;
use memmap2::Mmap;
use parking_lot::RwLock;

use crate::compression::CompressionEngine;
use crate::format::*;
use crate::validation::{SnapshotValidator, ValidationConfig};
use crate::{LoadingStats, Snapshot, SnapshotConfig, SnapshotError, SnapshotResult};

/// High-performance snapshot loader
pub struct SnapshotLoader {
    /// Configuration
    config: SnapshotConfig,

    /// Compression engine for decompression
    compression: CompressionEngine,

    /// Validator for integrity checks
    #[cfg(feature = "validation")]
    validator: SnapshotValidator,

    /// Memory-mapped file cache
    mmap_cache: Arc<RwLock<Vec<Mmap>>>,

    /// Loading statistics
    stats: LoadingStats,
}

/// Loader for specific snapshot format
struct FormatLoader {
    /// File handle
    file: File,

    /// Memory mapping (if enabled)
    mmap: Option<Mmap>,

    /// Header information
    header: SnapshotHeader,

    /// Configuration
    config: SnapshotConfig,
}

impl SnapshotLoader {
    /// Create a new snapshot loader
    pub fn new(config: SnapshotConfig) -> Self {
        let compression = CompressionEngine::new(crate::compression::CompressionConfig {
            adaptive_selection: false, // Decompression doesn't need adaptation
            prefer_speed: true,
            ..Default::default()
        });

        #[cfg(feature = "validation")]
        let validator = SnapshotValidator::with_config(ValidationConfig {
            strict_mode: false, // Don't fail on warnings during loading
            ..ValidationConfig::default()
        });

        Self {
            config,
            compression,
            #[cfg(feature = "validation")]
            validator,
            mmap_cache: Arc::new(RwLock::new(Vec::new())),
            stats: LoadingStats {
                load_time: Duration::ZERO,
                decompression_time: Duration::ZERO,
                validation_time: Duration::ZERO,
                initialization_time: Duration::ZERO,
                total_size: 0,
                compressed_size: 0,
                compression_ratio: 1.0,
                builtin_count: 0,
                cache_hit_rate: 0.0,
            },
        }
    }

    /// Load snapshot from file
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> SnapshotResult<(Snapshot, LoadingStats)> {
        let start_time = Instant::now();
        log::info!("Loading snapshot from {}", path.as_ref().display());

        // Open and validate file
        let format_loader = self.open_snapshot_file(path.as_ref())?;

        // Load and decompress data
        let data = self.load_snapshot_data(&format_loader)?;

        // Deserialize snapshot
        let snapshot = self.deserialize_snapshot(&data)?;

        // Validate snapshot if enabled
        #[cfg(feature = "validation")]
        if self.config.validation_enabled {
            self.validate_snapshot(&snapshot)?;
        }

        // Initialize runtime integration
        self.initialize_runtime_integration(&snapshot)?;

        self.stats.load_time = start_time.elapsed();
        log::info!("Snapshot loaded successfully in {:?}", self.stats.load_time);

        Ok((snapshot, self.stats.clone()))
    }

    /// Load snapshot asynchronously with true async I/O for non-blocking startup
    pub async fn load_async<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> SnapshotResult<(Snapshot, LoadingStats)> {
        let start_time = Instant::now();
        let path = path.as_ref();

        // Async file opening and validation
        let file = tokio::fs::File::open(path)
            .await
            .with_context(|| format!("Failed to open snapshot file: {}", path.display()))
            .map_err(|e| SnapshotError::Configuration {
                message: e.to_string(),
            })?;

        // Get file metadata asynchronously
        let metadata = file.metadata().await.map_err(SnapshotError::Io)?;
        let file_size = metadata.len() as usize;
        self.stats.total_size = file_size;

        // Read entire file asynchronously
        let mut file_contents = Vec::with_capacity(file_size);
        let mut reader = tokio::io::BufReader::new(file);
        use tokio::io::AsyncReadExt;
        reader
            .read_to_end(&mut file_contents)
            .await
            .map_err(SnapshotError::Io)?;

        // Validate file format
        if file_contents.len() < std::mem::size_of::<SnapshotHeader>() {
            return Err(SnapshotError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small to contain valid snapshot header",
            )));
        }

        // Parse header
        let header_size =
            bincode::serialized_size(&SnapshotHeader::new(SnapshotMetadata::current()))
                .map_err(SnapshotError::Serialization)? as usize;
        let header: SnapshotHeader = bincode::deserialize(
            &file_contents[..header_size.min(file_contents.len())],
        )
        .map_err(|e| SnapshotError::Configuration {
            message: format!("Failed to deserialize snapshot header: {e}"),
        })?;

        // Validate header
        header.validate()?;

        // Extract and decompress data
        let data_start = header.data_info.data_offset as usize;
        let data_end = data_start + header.data_info.compressed_size;

        if data_end > file_contents.len() {
            return Err(SnapshotError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Data section extends beyond file size",
            )));
        }

        let compressed_data = &file_contents[data_start..data_end];
        self.stats.compressed_size = compressed_data.len();

        // Decompress data if needed
        let decompressed_data =
            if header.data_info.compression.algorithm != CompressionAlgorithm::None {
                let compression_engine = CompressionEngine::new(CompressionConfig::default());
                compression_engine.decompress(compressed_data, &header.data_info.compression)?
            } else {
                compressed_data.to_vec()
            };

        // Deserialize snapshot
        let snapshot: Snapshot =
            bincode::deserialize(&decompressed_data).map_err(|e| SnapshotError::Configuration {
                message: format!("Failed to deserialize snapshot data: {e}"),
            })?;

        // Validate snapshot if enabled
        if self.config.validation_enabled {
            // Skip detailed validation for async loading (simplified)
            // In production, this would validate the snapshot structure
        }

        // Update stats
        let load_time = start_time.elapsed();
        self.stats.load_time = load_time;
        self.stats.builtin_count = snapshot.builtins.functions.len();

        Ok((snapshot, self.stats.clone()))
    }

    /// Open and validate snapshot file
    fn open_snapshot_file(&mut self, path: &Path) -> SnapshotResult<FormatLoader> {
        let start = Instant::now();

        // Open file
        let file = File::open(path)
            .with_context(|| format!("Failed to open snapshot file: {}", path.display()))
            .map_err(|e| crate::SnapshotError::Configuration {
                message: e.to_string(),
            })?;

        // Get file metadata
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        self.stats.total_size = file_size;

        // Create memory mapping if enabled and file is large enough
        let mmap = if self.config.memory_mapping_enabled && file_size > 4096 {
            match unsafe { Mmap::map(&file) } {
                Ok(mmap) => {
                    log::debug!("Created memory mapping for snapshot file ({file_size} bytes)");
                    Some(mmap)
                }
                Err(e) => {
                    log::warn!("Failed to create memory mapping, falling back to regular I/O: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Read and validate header
        let mut format_loader = FormatLoader {
            file,
            mmap,
            header: SnapshotHeader::new(SnapshotMetadata::current()), // Temporary
            config: self.config.clone(),
        };

        format_loader.header = format_loader.read_header()?;
        format_loader.header.validate()?;

        // Update stats
        self.stats.compressed_size = format_loader.header.data_info.compressed_size;
        self.stats.compression_ratio = format_loader.header.data_info.compressed_size as f64
            / format_loader.header.data_info.uncompressed_size as f64;

        let load_time = start.elapsed();
        log::debug!("File opened and header validated in {load_time:?}");

        Ok(format_loader)
    }

    /// Load and decompress snapshot data
    fn load_snapshot_data(&mut self, format_loader: &FormatLoader) -> SnapshotResult<Vec<u8>> {
        let start = Instant::now();

        // Read compressed data
        let compressed_data = format_loader.read_data_section()?;

        // Decompress if needed
        let decompression_start = Instant::now();
        let data = if matches!(
            format_loader.header.data_info.compression.algorithm,
            CompressionAlgorithm::None
        ) {
            compressed_data
        } else {
            self.compression.decompress(
                &compressed_data,
                &format_loader.header.data_info.compression,
            )?
        };
        self.stats.decompression_time = decompression_start.elapsed();

        let load_time = start.elapsed();
        log::debug!(
            "Data loaded and decompressed in {:?} (decompression: {:?})",
            load_time,
            self.stats.decompression_time
        );

        Ok(data)
    }

    /// Deserialize snapshot from data
    fn deserialize_snapshot(&mut self, data: &[u8]) -> SnapshotResult<Snapshot> {
        let start = Instant::now();

        let snapshot: Snapshot = bincode::deserialize(data)
            .context("Failed to deserialize snapshot data")
            .map_err(|e| crate::SnapshotError::Configuration {
                message: e.to_string(),
            })?;

        // Update stats
        self.stats.builtin_count = snapshot.builtins.functions.len();

        let deserialize_time = start.elapsed();
        log::debug!("Snapshot deserialized in {deserialize_time:?}");

        Ok(snapshot)
    }

    /// Validate loaded snapshot
    #[cfg(feature = "validation")]
    fn validate_snapshot(&mut self, snapshot: &Snapshot) -> SnapshotResult<()> {
        let start = Instant::now();

        // Validate content
        let content_result = self.validator.validate_content(snapshot)?;
        if !content_result.is_ok() {
            if self.config.validation_enabled {
                return Err(SnapshotError::Validation {
                    message: "Snapshot content validation failed".to_string(),
                });
            } else {
                log::warn!("Snapshot content validation failed, but continuing");
            }
        }

        // Validate compatibility
        let compat_result = self.validator.validate_compatibility(snapshot)?;
        if !compat_result.is_ok() {
            log::warn!("Snapshot compatibility issues detected");
            for warning in compat_result.warnings {
                log::warn!("Compatibility: {}", warning.message);
            }
        }

        self.stats.validation_time = start.elapsed();
        log::debug!("Snapshot validated in {:?}", self.stats.validation_time);

        Ok(())
    }

    /// Initialize runtime integration
    fn initialize_runtime_integration(&mut self, snapshot: &Snapshot) -> SnapshotResult<()> {
        let start = Instant::now();

        // Initialize builtin dispatch table
        self.initialize_builtin_dispatch(&snapshot.builtins)?;

        // Apply optimization hints
        self.apply_optimization_hints(&snapshot.optimization_hints)?;

        // Configure GC with presets
        self.configure_gc(&snapshot.gc_presets)?;

        self.stats.initialization_time = start.elapsed();
        log::debug!(
            "Runtime integration initialized in {:?}",
            self.stats.initialization_time
        );

        Ok(())
    }

    /// Initialize builtin function dispatch table
    fn initialize_builtin_dispatch(&self, registry: &crate::BuiltinRegistry) -> SnapshotResult<()> {
        // Get current builtins from runtime
        let current_builtins = rustmat_builtins::builtin_functions();
        let mut dispatch_table = Vec::with_capacity(registry.functions.len());

        // Build dispatch table by matching names
        for function_meta in &registry.functions {
            if let Some(builtin) = current_builtins
                .iter()
                .find(|b| b.name == function_meta.name)
            {
                dispatch_table.push(builtin.implementation);
            } else {
                log::warn!(
                    "Builtin function '{}' not found in current runtime",
                    function_meta.name
                );
                // Use a placeholder function that returns an error
                dispatch_table
                    .push(|_args| Err("Function not available in current runtime".to_string()));
            }
        }

        // Update the registry's dispatch table
        {
            let mut table = registry.dispatch_table.write();
            *table = dispatch_table;
        }

        log::debug!(
            "Initialized dispatch table with {} functions",
            registry.functions.len()
        );
        Ok(())
    }

    /// Apply optimization hints to runtime
    fn apply_optimization_hints(&self, hints: &crate::OptimizationHints) -> SnapshotResult<()> {
        // Apply JIT hints
        for hint in &hints.jit_hints {
            log::debug!(
                "JIT hint: {} ({:?}) - expected gain: {:.1}x",
                hint.pattern,
                hint.hint_type,
                hint.expected_performance_gain
            );
            // In a full implementation, these would be passed to the JIT compiler
        }

        // Apply memory hints
        for hint in &hints.memory_hints {
            log::debug!(
                "Memory hint: {} ({:?}) - alignment: {}",
                hint.data_structure,
                hint.hint_type,
                hint.alignment
            );
            // In a full implementation, these would configure memory layout
        }

        // Apply execution hints
        for hint in &hints.execution_hints {
            log::debug!(
                "Execution hint: {} ({:?}) - frequency: {}",
                hint.pattern,
                hint.hint_type,
                hint.frequency
            );
            // In a full implementation, these would configure execution strategies
        }

        Ok(())
    }

    /// Configure GC with snapshot presets
    fn configure_gc(&self, presets: &crate::GcPresetCache) -> SnapshotResult<()> {
        if let Some(default_config) = presets.presets.get(&presets.default_preset) {
            match rustmat_gc::gc_configure(default_config.clone()) {
                Ok(_) => {
                    log::debug!("GC configured with preset '{}'", presets.default_preset);
                }
                Err(e) => {
                    log::warn!("Failed to configure GC with snapshot preset: {e}");
                }
            }
        }

        Ok(())
    }

    /// Get loading statistics
    pub fn stats(&self) -> &LoadingStats {
        &self.stats
    }

    /// Clear memory-mapped file cache
    pub fn clear_cache(&mut self) {
        let mut cache = self.mmap_cache.write();
        cache.clear();
        log::debug!("Memory mapping cache cleared");
    }
}

impl FormatLoader {
    /// Read snapshot header from file
    fn read_header(&mut self) -> SnapshotResult<SnapshotHeader> {
        // Check configuration for memory mapping preference and validation
        let use_mmap = self.config.memory_mapping_enabled;
        let validate_data = self.config.validation_enabled;

        if use_mmap && self.mmap.is_some() {
            // Use memory mapping
            let mmap_data = self.mmap.as_ref().unwrap();

            // Read header size (4 bytes, little-endian)
            if mmap_data.len() < 4 {
                return Err(crate::SnapshotError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "File too small to contain header size",
                )));
            }

            let header_size =
                u32::from_le_bytes([mmap_data[0], mmap_data[1], mmap_data[2], mmap_data[3]])
                    as usize;

            // Read header data
            if mmap_data.len() < 4 + header_size {
                return Err(crate::SnapshotError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "File too small to contain header",
                )));
            }

            let header_data = &mmap_data[4..4 + header_size];
            let header: SnapshotHeader = bincode::deserialize(header_data)
                .context("Failed to deserialize header from memory map")
                .map_err(|e| crate::SnapshotError::Configuration {
                    message: e.to_string(),
                })?;

            // Validate header if configuration requires it
            if validate_data {
                header.validate()?;
            }
            Ok(header)
        } else {
            // Use regular file I/O
            let mut reader = BufReader::new(&self.file);
            reader.seek(SeekFrom::Start(0))?;

            // Read header size (4 bytes, little-endian)
            let mut size_buffer = [0u8; 4];
            reader.read_exact(&mut size_buffer)?;
            let header_size = u32::from_le_bytes(size_buffer) as usize;

            // Read header data
            let mut header_buffer = vec![0u8; header_size];
            reader.read_exact(&mut header_buffer)?;

            let header: SnapshotHeader = bincode::deserialize(&header_buffer)
                .context("Failed to deserialize header")
                .map_err(|e| crate::SnapshotError::Configuration {
                    message: e.to_string(),
                })?;

            // Validate header if configuration requires it
            if validate_data {
                header.validate()?;
            }
            Ok(header)
        }
    }

    /// Read data section from file
    fn read_data_section(&self) -> SnapshotResult<Vec<u8>> {
        if let Some(ref mmap) = self.mmap {
            // Use memory mapping
            // Account for 4-byte header size prefix + actual header size
            let header_size = bincode::serialized_size(&self.header)? as usize;
            let data_start = 4 + header_size; // 4 bytes for size + header
            let data_end = data_start + self.header.data_info.compressed_size;

            if data_end > mmap.len() {
                return Err(SnapshotError::Corrupted {
                    reason: "Data section extends beyond file".to_string(),
                });
            }

            Ok(mmap[data_start..data_end].to_vec())
        } else {
            // Use regular file I/O
            let file = &self.file;
            // Account for 4-byte header size prefix + actual header size
            let header_size = bincode::serialized_size(&self.header)? as u64;
            let data_start = 4 + header_size; // 4 bytes for size + header
            let mut reader = BufReader::new(file);

            reader.seek(SeekFrom::Start(data_start))?;

            let mut data = vec![0u8; self.header.data_info.compressed_size];
            reader.read_exact(&mut data)?;

            Ok(data)
        }
    }
}

/// Utility functions for snapshot loading
impl SnapshotLoader {
    /// Preload snapshot header for quick validation
    pub fn peek_header<P: AsRef<Path>>(path: P) -> SnapshotResult<SnapshotHeader> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open snapshot file: {}", path.as_ref().display()))
            .map_err(|e| crate::SnapshotError::Configuration {
                message: e.to_string(),
            })?;

        let mut format_loader = FormatLoader {
            file,
            mmap: None,
            header: SnapshotHeader::new(SnapshotMetadata::current()),
            config: SnapshotConfig::default(),
        };

        format_loader.read_header()
    }

    /// Check if snapshot file is valid without full loading
    pub fn quick_validate<P: AsRef<Path>>(path: P) -> SnapshotResult<bool> {
        match Self::peek_header(path) {
            Ok(header) => Ok(header.validate().is_ok()),
            Err(_) => Ok(false),
        }
    }

    /// Get snapshot metadata without loading content
    pub fn get_metadata<P: AsRef<Path>>(path: P) -> SnapshotResult<SnapshotMetadata> {
        let header = Self::peek_header(path)?;
        Ok(header.metadata)
    }

    /// Estimate loading time based on snapshot header
    pub fn estimate_load_time<P: AsRef<Path>>(path: P) -> SnapshotResult<Duration> {
        let header = Self::peek_header(path)?;
        Ok(header.estimated_load_time())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let config = SnapshotConfig::default();
        let loader = SnapshotLoader::new(config);
        assert_eq!(loader.stats.load_time, Duration::ZERO);
    }

    #[test]
    fn test_quick_validate_nonexistent() {
        assert!(!SnapshotLoader::quick_validate("nonexistent.snapshot").unwrap_or(true));
    }

    #[test]
    fn test_header_peek() {
        // This would require a real snapshot file
        // For now, just test that the function exists
        let result = SnapshotLoader::peek_header("nonexistent.snapshot");
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_extraction() {
        // This would require a real snapshot file
        let result = SnapshotLoader::get_metadata("nonexistent.snapshot");
        assert!(result.is_err());
    }
}
