#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePlottingMode {
    Auto,
    Interactive,
    Static,
}

#[cfg(feature = "gui")]
pub fn set_runtime_plotting_mode(mode: RuntimePlottingMode) {
    let mapped = match mode {
        RuntimePlottingMode::Auto => super::engine::native::RuntimePlottingMode::Auto,
        RuntimePlottingMode::Interactive => super::engine::native::RuntimePlottingMode::Interactive,
        RuntimePlottingMode::Static => super::engine::native::RuntimePlottingMode::Static,
    };
    super::engine::native::set_runtime_plotting_mode(mapped);
}

#[cfg(not(feature = "gui"))]
pub fn set_runtime_plotting_mode(_mode: RuntimePlottingMode) {}
