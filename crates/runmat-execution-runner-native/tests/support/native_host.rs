//! Match the CLI's command stack for harness-free native host witnesses.
const HOST_STACK_BYTES: usize = 64 * 1024 * 1024;

pub fn run(entry: fn()) {
    let host = std::thread::Builder::new()
        .name("native-conformance-host".into())
        .stack_size(HOST_STACK_BYTES)
        .spawn(entry)
        .expect("start native conformance host");
    host.join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
