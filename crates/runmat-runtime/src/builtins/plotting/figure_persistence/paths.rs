use super::*;

pub(super) fn parse_save_format(raw: &str, builtin: &'static str) -> BuiltinResult<SaveFormat> {
    let lowered = raw
        .trim()
        .trim_start_matches("-d")
        .trim_start_matches('.')
        .to_ascii_lowercase();
    match lowered.as_str() {
        "fig" => Ok(SaveFormat::Fig),
        "png" => Ok(SaveFormat::Png),
        "m" | "mfig" | "jpeg" | "jpg" | "tiff" | "tif" | "pdf" | "eps" | "epsc" | "svg" => {
            Err(persistence_error(
                &ERROR_UNSUPPORTED_FORMAT,
                builtin,
                format!("{builtin}: format '{lowered}' is not available yet; RunMat currently supports FIG and PNG"),
            ))
        }
        "" => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: format must not be empty"),
        )),
        other => Err(persistence_error(
            &ERROR_UNSUPPORTED_FORMAT,
            builtin,
            format!("{builtin}: unsupported output format '{other}'"),
        )),
    }
}

pub(super) fn infer_save_format(
    filename: &str,
    builtin: &'static str,
) -> BuiltinResult<Option<SaveFormat>> {
    match Path::new(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("fig") => Ok(Some(SaveFormat::Fig)),
        Some("png") => Ok(Some(SaveFormat::Png)),
        Some(ext) => parse_save_format(ext, builtin).map(Some),
        None => Ok(None),
    }
}

pub(super) fn output_path_for_format(
    filename: &str,
    format: SaveFormat,
    builtin: &'static str,
) -> BuiltinResult<PathBuf> {
    let path = Path::new(filename);
    if path.extension().is_some() {
        return Ok(path.to_path_buf());
    }
    append_extension(path, format.standard_extension(), builtin)
}

pub(super) fn fig_path(
    filename: &str,
    reject_non_fig: bool,
    builtin: &'static str,
) -> BuiltinResult<PathBuf> {
    let path = Path::new(filename);
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("fig") => Ok(path.to_path_buf()),
        Some(ext) if reject_non_fig => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected a .fig filename, got extension '.{ext}'"),
        )),
        Some(_) => Ok(path.to_path_buf()),
        None => append_extension(path, "fig", builtin),
    }
}

pub(super) fn append_extension(
    path: &Path,
    ext: &str,
    builtin: &'static str,
) -> BuiltinResult<PathBuf> {
    let Some(file_name) = path.file_name() else {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename must include a file name before appending '.{ext}'"),
        ));
    };
    let mut file_name = OsString::from(file_name);
    file_name.push(".");
    file_name.push(ext);
    Ok(path.with_file_name(file_name))
}
