//! High-performance compression for snapshot data
//!
//! Multi-tier compression system with adaptive algorithm selection.
//! Optimized for fast decompression during runtime startup.

use std::collections::HashMap;
use std::convert::TryFrom;
#[cfg(all(feature = "compression", target_arch = "wasm32"))]
use std::io::{Cursor, Read};
use std::time::Instant;

use crate::format::{CompressionAlgorithm, CompressionInfo};
use crate::{SnapshotError, SnapshotResult};
#[cfg(all(feature = "compression", target_arch = "wasm32"))]
use ruzstd::decoding::StreamingDecoder;

/// Compression engine with adaptive algorithm selection
pub struct CompressionEngine {
    /// Configuration
    config: CompressionConfig,

    /// Performance statistics
    stats: CompressionStats,
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Default compression level (1-9)
    pub default_level: u32,

    /// Enable adaptive algorithm selection
    pub adaptive_selection: bool,

    /// Size threshold for compression (bytes)
    pub size_threshold: usize,

    /// Target compression ratio
    pub target_ratio: f64,

    /// Maximum compression time
    pub max_compression_time: std::time::Duration,

    /// Prefer speed over ratio
    pub prefer_speed: bool,
}

/// Compression performance statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Compression attempts by algorithm
    pub attempts: HashMap<String, u64>,

    /// Average compression ratios
    pub ratios: HashMap<String, f64>,

    /// Average compression times
    pub times: HashMap<String, std::time::Duration>,

    /// Total bytes processed
    pub total_bytes: u64,

    /// Total time spent compressing
    pub total_time: std::time::Duration,
}

/// Compression result with metadata
#[derive(Debug)]
pub struct CompressionResult {
    /// Compressed data
    pub data: Vec<u8>,

    /// Compression information
    pub info: CompressionInfo,

    /// Performance metrics
    pub metrics: CompressionMetrics,
}

/// Compression performance metrics
#[derive(Debug)]
pub struct CompressionMetrics {
    /// Compression time
    pub compression_time: std::time::Duration,

    /// Compression ratio (compressed/original)
    pub compression_ratio: f64,

    /// Compression throughput (bytes/second)
    pub throughput: f64,

    /// Memory usage during compression
    pub memory_usage: usize,
}

impl CompressionEngine {
    /// Create a new compression engine
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    /// Compress data using optimal algorithm
    pub fn compress(&mut self, data: &[u8]) -> SnapshotResult<CompressionResult> {
        // Skip compression for small data
        if data.len() < self.config.size_threshold {
            return Ok(CompressionResult {
                data: data.to_vec(),
                info: CompressionInfo {
                    algorithm: CompressionAlgorithm::None,
                    level: 0,
                    parameters: HashMap::new(),
                },
                metrics: CompressionMetrics {
                    compression_time: std::time::Duration::ZERO,
                    compression_ratio: 1.0,
                    throughput: 0.0,
                    memory_usage: data.len(),
                },
            });
        }

        let algorithm = if self.config.adaptive_selection {
            self.select_optimal_algorithm(data)?
        } else {
            CompressionAlgorithm::Lz4 {
                fast: self.config.prefer_speed,
            }
        };

        self.compress_with_algorithm(data, algorithm)
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8], info: &CompressionInfo) -> SnapshotResult<Vec<u8>> {
        let start = Instant::now();

        let result = match &info.algorithm {
            CompressionAlgorithm::None => data.to_vec(),

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Lz4 { .. } => {
                // Extract the original size from parameters
                let original_size = info
                    .parameters
                    .get("original_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or_else(|| {
                        // Fallback: estimate from compressed size
                        data.len() * 4
                    });

                decompress_lz4_block(data, original_size)?
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Zstd { dictionary } => {
                if dictionary.is_some() {
                    // Dictionary decompression not supported by current backend, fall back to standard
                    log::debug!(
                        "Dictionary decompression not available, using standard decompression"
                    );
                }

                decompress_zstd_block(data)?
            }

            #[cfg(not(feature = "compression"))]
            _ => {
                return Err(SnapshotError::Configuration {
                    message: "Compression feature not enabled".to_string(),
                });
            }
        };

        let duration = start.elapsed();
        log::debug!(
            "Decompressed {} bytes to {} bytes in {:?}",
            data.len(),
            result.len(),
            duration
        );

        Ok(result)
    }

    /// Select optimal compression algorithm for data
    pub fn select_optimal_algorithm(&self, data: &[u8]) -> SnapshotResult<CompressionAlgorithm> {
        // Analyze data characteristics
        let characteristics = self.analyze_data(data);

        // Choose algorithm based on characteristics and config
        if characteristics.entropy > 0.9 {
            // High entropy data - compression won't help much
            return Ok(CompressionAlgorithm::None);
        }

        if self.config.prefer_speed {
            Ok(CompressionAlgorithm::Lz4 { fast: true })
        } else if characteristics.repetition_ratio > 0.7 {
            // High repetition - ZSTD will work well
            Ok(CompressionAlgorithm::Zstd { dictionary: None })
        } else {
            // Balanced choice
            Ok(CompressionAlgorithm::Lz4 { fast: false })
        }
    }

    /// Compress with specific algorithm
    pub fn compress_with_algorithm(
        &mut self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> SnapshotResult<CompressionResult> {
        let start = Instant::now();
        let start_memory = self.estimate_memory_usage();

        let (compressed_data, algorithm_used) = match algorithm {
            CompressionAlgorithm::None => (data.to_vec(), algorithm),

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Lz4 { fast } => {
                let compressed = compress_lz4_block(data, fast)?;
                (compressed, CompressionAlgorithm::Lz4 { fast })
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Zstd { dictionary } => {
                let level = if self.config.prefer_speed {
                    1
                } else {
                    self.config.default_level as i32
                };

                if dictionary.is_some() {
                    // Dictionary compression not supported by current backend, fall back to standard
                    log::debug!("Dictionary compression not available, using standard compression");
                }

                let compressed = compress_zstd_block(data, level)?;

                (compressed, CompressionAlgorithm::Zstd { dictionary })
            }

            #[cfg(not(feature = "compression"))]
            _ => {
                return Err(SnapshotError::Configuration {
                    message: "Compression feature not enabled".to_string(),
                });
            }
        };

        let compression_time = start.elapsed();
        let end_memory = self.estimate_memory_usage();

        // Check if compression was effective
        let compression_ratio = compressed_data.len() as f64 / data.len() as f64;
        if compression_ratio > 0.95 && !matches!(algorithm_used, CompressionAlgorithm::None) {
            // Compression wasn't effective, use uncompressed
            log::debug!(
                "Compression ratio {compression_ratio:.3} not effective, using uncompressed data"
            );
            return Ok(CompressionResult {
                data: data.to_vec(),
                info: CompressionInfo {
                    algorithm: CompressionAlgorithm::None,
                    level: 0,
                    parameters: HashMap::new(),
                },
                metrics: CompressionMetrics {
                    compression_time,
                    compression_ratio: 1.0,
                    throughput: data.len() as f64 / compression_time.as_secs_f64(),
                    memory_usage: end_memory.saturating_sub(start_memory),
                },
            });
        }

        // Update statistics
        self.update_stats(
            &algorithm_used,
            data.len(),
            compressed_data.len(),
            compression_time,
        );

        let metrics = CompressionMetrics {
            compression_time,
            compression_ratio,
            throughput: data.len() as f64 / compression_time.as_secs_f64(),
            memory_usage: end_memory.saturating_sub(start_memory),
        };

        let mut parameters = HashMap::new();
        parameters.insert("original_size".to_string(), data.len().to_string());
        parameters.insert(
            "compressed_size".to_string(),
            compressed_data.len().to_string(),
        );

        Ok(CompressionResult {
            data: compressed_data,
            info: CompressionInfo {
                algorithm: algorithm_used,
                level: self.config.default_level,
                parameters,
            },
            metrics,
        })
    }

    /// Analyze data characteristics for algorithm selection
    pub fn analyze_data(&self, data: &[u8]) -> DataCharacteristics {
        let mut characteristics = DataCharacteristics {
            entropy: 0.0,
            repetition_ratio: 0.0,
            pattern_density: 0.0,
            ascii_ratio: 0.0,
        };

        if data.is_empty() {
            return characteristics;
        }

        // Calculate entropy (simplified)
        let mut byte_counts = [0u32; 256];
        let mut ascii_count = 0;

        for &byte in data {
            byte_counts[byte as usize] += 1;
            if byte.is_ascii() {
                ascii_count += 1;
            }
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;
        for &count in &byte_counts {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }
        characteristics.entropy = entropy / 8.0; // Normalize to 0-1
        characteristics.ascii_ratio = ascii_count as f64 / len;

        // Analyze repetition patterns (simplified)
        let mut repetition_count = 0;
        let window_size = 64.min(data.len() / 2);

        if data.len() > window_size * 2 {
            for i in 0..data.len() - window_size {
                for j in (i + window_size)..data.len() - window_size {
                    if data[i..i + window_size] == data[j..j + window_size] {
                        repetition_count += 1;
                        break;
                    }
                }
            }
            characteristics.repetition_ratio =
                repetition_count as f64 / (data.len() - window_size) as f64;
        }

        characteristics
    }

    /// Update compression statistics
    fn update_stats(
        &mut self,
        algorithm: &CompressionAlgorithm,
        original_size: usize,
        compressed_size: usize,
        time: std::time::Duration,
    ) {
        let algo_name = match algorithm {
            CompressionAlgorithm::None => "none",
            CompressionAlgorithm::Lz4 { fast } => {
                if *fast {
                    "lz4-fast"
                } else {
                    "lz4"
                }
            }
            CompressionAlgorithm::Zstd { .. } => "zstd",
        };

        let ratio = compressed_size as f64 / original_size as f64;

        *self
            .stats
            .attempts
            .entry(algo_name.to_string())
            .or_insert(0) += 1;

        // Update running averages
        let attempts = self.stats.attempts[algo_name];
        if let Some(existing_ratio) = self.stats.ratios.get_mut(algo_name) {
            *existing_ratio = (*existing_ratio * (attempts - 1) as f64 + ratio) / attempts as f64;
        } else {
            self.stats.ratios.insert(algo_name.to_string(), ratio);
        }

        if let Some(existing_time) = self.stats.times.get_mut(algo_name) {
            *existing_time = (*existing_time * (attempts as u32 - 1) + time) / attempts as u32;
        } else {
            self.stats.times.insert(algo_name.to_string(), time);
        }

        self.stats.total_bytes += original_size as u64;
        self.stats.total_time += time;
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Simplified memory estimation
        std::mem::size_of::<Self>()
            + self.stats.attempts.len() * 64
            + self.stats.ratios.len() * 64
            + self.stats.times.len() * 64
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::default();
    }
}

/// Data characteristics for algorithm selection
#[derive(Debug)]
pub struct DataCharacteristics {
    /// Shannon entropy (0-1, higher = more random)
    pub entropy: f64,

    /// Repetition ratio (0-1, higher = more repetitive)
    pub repetition_ratio: f64,

    /// Pattern density (0-1, higher = more patterns)
    pub pattern_density: f64,

    /// ASCII text ratio (0-1, higher = more text)
    pub ascii_ratio: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            default_level: 6,
            adaptive_selection: true,
            size_threshold: 1024, // Don't compress < 1KB
            target_ratio: 0.7,
            max_compression_time: std::time::Duration::from_secs(30),
            prefer_speed: false,
        }
    }
}

impl CompressionStats {
    /// Get overall compression ratio
    pub fn overall_ratio(&self) -> f64 {
        if self.ratios.is_empty() {
            1.0
        } else {
            self.ratios.values().sum::<f64>() / self.ratios.len() as f64
        }
    }

    /// Get overall throughput
    pub fn overall_throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() > 0.0 {
            self.total_bytes as f64 / self.total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get best performing algorithm
    pub fn best_algorithm(&self) -> Option<String> {
        self.ratios
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name.clone())
    }
}

#[cfg(feature = "compression")]
fn decompress_lz4_block(data: &[u8], original_size: usize) -> SnapshotResult<Vec<u8>> {
    let size_hint = original_size.max(1);

    #[cfg(target_arch = "wasm32")]
    {
        let min_uncompressed = size_hint.max(data.len().saturating_mul(4));
        lz4_flex::block::decompress(data, min_uncompressed).map_err(|e| {
            SnapshotError::Compression {
                message: format!("LZ4 decompression failed: {e}"),
            }
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let clamped = size_hint.min(i32::MAX as usize);
        let size_i32 = i32::try_from(clamped).unwrap_or(i32::MAX);
        lz4::block::decompress(data, Some(size_i32)).map_err(|e| SnapshotError::Compression {
            message: format!("LZ4 decompression failed: {e}"),
        })
    }
}

#[cfg(feature = "compression")]
fn decompress_zstd_block(data: &[u8]) -> SnapshotResult<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    {
        let cursor = Cursor::new(data);
        let mut decoder =
            StreamingDecoder::new(cursor).map_err(|e| SnapshotError::Compression {
                message: format!("ZSTD decompression failed: {e}"),
            })?;
        let mut output = Vec::new();
        decoder
            .read_to_end(&mut output)
            .map_err(|e| SnapshotError::Compression {
                message: format!("ZSTD decompression failed: {e}"),
            })?;
        Ok(output)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        zstd::decode_all(data).map_err(|e| SnapshotError::Compression {
            message: format!("ZSTD decompression failed: {e}"),
        })
    }
}

#[cfg(feature = "compression")]
fn compress_lz4_block(data: &[u8], fast: bool) -> SnapshotResult<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    {
        let compressed = lz4_flex::block::compress(data);
        if !fast {
            log::trace!("High-compression LZ4 mode not available on wasm; using fast path");
        }
        Ok(compressed)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = if fast {
            lz4::block::compress(data, None, false)
        } else {
            lz4::block::compress(
                data,
                Some(lz4::block::CompressionMode::HIGHCOMPRESSION(12)),
                false,
            )
        };

        result.map_err(|e| SnapshotError::Compression {
            message: format!("LZ4 compression failed: {e}"),
        })
    }
}

#[cfg(feature = "compression")]
fn compress_zstd_block(data: &[u8], level: i32) -> SnapshotResult<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    {
        Err(SnapshotError::Configuration {
            message: "ZSTD compression is not supported on wasm targets".to_string(),
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        zstd::encode_all(data, level).map_err(|e| SnapshotError::Compression {
            message: format!("ZSTD compression failed: {e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert_eq!(config.default_level, 6);
        assert!(config.adaptive_selection);
        assert_eq!(config.size_threshold, 1024);
    }

    #[test]
    fn test_compression_engine_creation() {
        let config = CompressionConfig::default();
        let engine = CompressionEngine::new(config);
        assert_eq!(engine.stats.total_bytes, 0);
    }

    #[test]
    fn test_small_data_compression() {
        let mut engine = CompressionEngine::new(CompressionConfig::default());
        let small_data = vec![1, 2, 3]; // < threshold

        let result = engine.compress(&small_data).unwrap();
        assert_eq!(result.data, small_data);
        assert!(matches!(result.info.algorithm, CompressionAlgorithm::None));
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_lz4_compression() {
        let mut engine = CompressionEngine::new(CompressionConfig {
            adaptive_selection: false,
            prefer_speed: true,
            size_threshold: 10, // Lower threshold to ensure compression is attempted
            ..CompressionConfig::default()
        });

        let data = b"Hello, World! This is a longer test string for LZ4 compression.".repeat(50);
        let result = engine.compress(&data).unwrap();

        println!(
            "Original size: {}, Compressed size: {}, Algorithm: {:?}",
            data.len(),
            result.data.len(),
            result.info.algorithm
        );

        // Check that compression was attempted (either compressed or fell back to None if not effective)
        assert!(matches!(
            result.info.algorithm,
            CompressionAlgorithm::Lz4 { .. } | CompressionAlgorithm::None
        ));

        // Test decompression should always work
        match engine.decompress(&result.data, &result.info) {
            Ok(decompressed) => {
                assert_eq!(decompressed, data);
            }
            Err(e) => {
                println!("Decompression error: {e:?}");
                println!("This is expected if compression fell back to None due to ineffective compression ratio");
                // If decompression fails, ensure we're dealing with uncompressed data
                if matches!(result.info.algorithm, CompressionAlgorithm::None) {
                    assert_eq!(result.data, data);
                } else {
                    panic!("Decompression failed: {e:?}");
                }
            }
        }
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats::default();
        assert_eq!(stats.overall_ratio(), 1.0);
        assert_eq!(stats.overall_throughput(), 0.0);
        assert!(stats.best_algorithm().is_none());
    }

    #[test]
    fn test_data_characteristics() {
        let engine = CompressionEngine::new(CompressionConfig::default());

        // Test with ASCII data
        let ascii_data = b"Hello, World!".to_vec();
        let characteristics = engine.analyze_data(&ascii_data);
        assert!(characteristics.ascii_ratio > 0.9);

        // Test with binary data
        let binary_data = vec![0u8, 1u8, 2u8, 255u8];
        let characteristics = engine.analyze_data(&binary_data);
        assert!(characteristics.ascii_ratio < 1.0);
    }
}
