use runmat_builtins::documentation_export::export_from_directory;
use std::{env, fs, path::Path};
#[path = "export_builtin_documentation/cli.rs"]
mod cli;
use cli::parse_options;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options(env::args_os().skip(1))?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let legacy_directory = repository.join("docs/builtins/reference");
    let export = export_from_directory(&legacy_directory, options.mode, &options.identities)?;
    if options.check_only {
        return Ok(());
    }
    let encoded = format!("{}\n", serde_json::to_string_pretty(&export)?);
    if let Some(path) = options.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, encoded)?;
    } else {
        print!("{encoded}");
    }
    Ok(())
}
