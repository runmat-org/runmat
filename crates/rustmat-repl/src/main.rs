use std::io::{self, Write};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut line = String::new();

    loop {
        line.clear();
        if stdout.write_all(b"> ").is_err() || stdout.flush().is_err() {
            break;
        }
        if stdin.read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let output = rustmat_repl::format_tokens(line.trim_end());
        if writeln!(stdout, "{}", output).is_err() {
            break;
        }
    }
}
