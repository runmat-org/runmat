use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let destination = destination(std::env::args_os().skip(1).collect())?;
    if matches!(destination, Destination::Help) {
        println!("Usage: export_builtin_migration_inventory [--output PATH]");
        return Ok(());
    }

    let inventory = runmat_runtime::builtin::migration_inventory::migration_inventory_json()?;
    match destination {
        Destination::Stdout => io::stdout().write_all(inventory.as_bytes())?,
        Destination::File(path) => OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(inventory.as_bytes())?,
        Destination::Help => unreachable!("help returned before inventory generation"),
    }
    Ok(())
}

enum Destination {
    Stdout,
    File(PathBuf),
    Help,
}

fn destination(arguments: Vec<std::ffi::OsString>) -> Result<Destination, &'static str> {
    match arguments.as_slice() {
        [] => Ok(Destination::Stdout),
        [flag] if flag == OsStr::new("--help") || flag == OsStr::new("-h") => Ok(Destination::Help),
        [flag, path] if flag == OsStr::new("--output") && !path.is_empty() => {
            Ok(Destination::File(PathBuf::from(path)))
        }
        _ => Err("expected no arguments, --help, or --output PATH"),
    }
}

#[cfg(test)]
mod tests {
    use super::{destination, Destination};
    use std::ffi::OsString;

    #[test]
    fn parses_closed_destination_contract() {
        assert!(matches!(destination(vec![]).unwrap(), Destination::Stdout));
        assert!(matches!(
            destination(vec![OsString::from("--help")]).unwrap(),
            Destination::Help
        ));
        assert!(matches!(
            destination(vec![
                OsString::from("--output"),
                OsString::from("inventory.json")
            ])
            .unwrap(),
            Destination::File(_)
        ));
        assert!(destination(vec![OsString::from("--unknown")]).is_err());
        assert!(destination(vec![OsString::from("--output")]).is_err());
    }
}
