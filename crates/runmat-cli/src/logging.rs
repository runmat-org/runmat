use std::io::Write;

pub fn format_log_record(
    buf: &mut env_logger::fmt::Formatter,
    record: &log::Record,
) -> std::io::Result<()> {
    let timestamp = buf.timestamp_nanos();
    writeln!(
        buf,
        "[{} {:>5} {}] {}",
        timestamp,
        record.level(),
        record.target(),
        record.args()
    )
}
