use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AotOptLevel {
    None,
    Size,
    Speed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AotPolicy {
    NativeSpecialized,
    ClosedWorld,
    DynamicRuntime,
    Portable,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum ColorMode {
    /// Detect color support for each output stream
    #[default]
    Auto,
    /// Always style human-readable output
    Always,
    /// Never emit CLI-owned ANSI styling
    Never,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum MeshElementOrderArg {
    Tet4,
    Tet10,
}

#[derive(Clone, Debug)]
pub struct FigureSize {
    pub width: u32,
    pub height: u32,
}
