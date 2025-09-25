fn main() {
    if let Err(err) = runmatfunc::run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
