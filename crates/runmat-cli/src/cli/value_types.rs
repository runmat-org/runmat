use clap::ValueEnum;
use runmat_gc::GcConfig;

#[derive(Clone, Debug, ValueEnum)]
pub enum CompressionAlg {
    /// No compression
    None,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstd compression (balanced)
    Zstd,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum OptLevel {
    /// No optimization
    None,
    /// Minimal optimization
    Size,
    /// Balanced optimization (default)
    Speed,
    /// Maximum optimization
    Aggressive,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum GcPreset {
    /// Minimize pause times
    LowLatency,
    /// Maximize throughput
    HighThroughput,
    /// Minimize memory usage
    LowMemory,
    /// Debug and analysis mode
    Debug,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum CaptureFiguresMode {
    Off,
    Auto,
    On,
}

#[derive(Clone, Debug)]
pub struct FigureSize {
    pub width: u32,
    pub height: u32,
}

impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

impl From<GcPreset> for GcConfig {
    fn from(preset: GcPreset) -> Self {
        match preset {
            GcPreset::LowLatency => GcConfig::low_latency(),
            GcPreset::HighThroughput => GcConfig::high_throughput(),
            GcPreset::LowMemory => GcConfig::low_memory(),
            GcPreset::Debug => GcConfig::debug(),
        }
    }
}

impl From<CompressionAlg> for runmat_snapshot::CompressionAlgorithm {
    fn from(alg: CompressionAlg) -> Self {
        use runmat_snapshot::CompressionAlgorithm;

        match alg {
            CompressionAlg::None => CompressionAlgorithm::None,
            CompressionAlg::Lz4 => CompressionAlgorithm::Lz4,
            CompressionAlg::Zstd => CompressionAlgorithm::Zstd,
        }
    }
}
