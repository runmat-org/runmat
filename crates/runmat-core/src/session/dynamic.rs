pub(super) fn path_matches_clear_name(path: &std::path::Path, name: &str) -> bool {
    let normalized = name.replace(['/', '\\'], std::path::MAIN_SEPARATOR_STR);
    let requested = std::path::Path::new(&normalized).with_extension("");
    if requested.components().count() > 1 {
        return path.with_extension("").ends_with(requested);
    }
    let Some(candidate) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let Some(requested) = requested.file_name().and_then(|stem| stem.to_str()) else {
        return false;
    };
    if cfg!(target_os = "windows") {
        candidate.eq_ignore_ascii_case(requested)
    } else {
        candidate == requested
    }
}

#[cfg(test)]
mod tests {
    use super::path_matches_clear_name;

    #[test]
    fn clear_names_match_stems_and_qualified_paths() {
        assert!(path_matches_clear_name(
            std::path::Path::new("project/lib/helper.m"),
            "helper"
        ));
        assert!(path_matches_clear_name(
            std::path::Path::new("project/lib/helper.m"),
            "lib/helper"
        ));
        assert!(!path_matches_clear_name(
            std::path::Path::new("project/other/helper.m"),
            "lib/helper"
        ));
    }
}
