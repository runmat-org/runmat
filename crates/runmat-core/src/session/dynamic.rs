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

#[path = "dynamic/product.rs"]
mod product;
pub(in crate::session) use product::{
    attach_dynamic_function_origins, dynamic_executable_revision, dynamic_pending_closure_revision,
};
use product::{dynamic_product_dependencies, reusable_dynamic_product};

#[path = "dynamic/completion.rs"]
mod completion;
use completion::finish_dynamic_invocation;

#[path = "dynamic/constructor.rs"]
mod constructor;

#[path = "dynamic/current_class_product.rs"]
mod current_class_product;
pub(super) use current_class_product::resolve as resolve_current_class_product;

#[path = "dynamic/loader.rs"]
mod loader;
pub(super) use loader::{load_dynamic_function, DynamicFunctionEnvironment};

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
