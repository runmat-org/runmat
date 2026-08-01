use std::io::Write;

pub fn format_log_record(
    buf: &mut env_logger::fmt::Formatter,
    record: &log::Record,
) -> std::io::Result<()> {
    let timestamp = buf.timestamp_nanos();
    let styles = crate::presentation::stderr();
    let level = match record.level() {
        log::Level::Error => styles.error(record.level()),
        log::Level::Warn => styles.warning(record.level()),
        log::Level::Info => styles.info(record.level()),
        log::Level::Debug => styles.identifier(record.level()),
        log::Level::Trace => styles.muted(record.level()),
    };
    writeln!(
        buf,
        "[{} {:>5} {}] {}",
        styles.muted(timestamp),
        level,
        styles.muted(record.target()),
        record.args()
    )
}
