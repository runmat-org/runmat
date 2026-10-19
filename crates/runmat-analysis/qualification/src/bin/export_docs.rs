use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new("--out")) {
        return Err("Usage: export_docs --out <file>".into());
    }
    let path = PathBuf::from(arguments.next().ok_or("Missing output path")?);
    if arguments.next().is_some() {
        return Err("Unexpected argument".into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::new();
    runmat_analysis_qualification::export::write_requirements(&mut bytes)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}
