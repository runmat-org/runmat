//! Opt-in diagnostics for standalone integration-test processes.
//! Observation only: this does not change fusion or placement policy.
pub(crate) fn enable() {
    if std::env::var_os("RUNMAT_DEBUG_FUSION").is_none() {
        return;
    }
    static LOGGER: FusionLogger = FusionLogger;
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Debug);
}

struct FusionLogger;
impl log::Log for FusionLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
            && (metadata
                .target()
                .starts_with("runmat_accelerate::fusion_exec")
                || metadata.target().starts_with("runmat_vm::accel::fusion"))
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.target(), record.args());
        }
    }
    fn flush(&self) {}
}
