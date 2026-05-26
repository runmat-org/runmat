use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
            continue;
        }
        if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn pre_test_source(source: &str) -> &str {
    source.split("#[cfg(test)]").next().unwrap_or(source)
}

#[test]
fn migrated_builtin_files_do_not_duplicate_stable_error_constants() {
    let builtins_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/builtins");
    let mut files = Vec::new();
    collect_rs_files(&builtins_root, &mut files);

    for file in files {
        let Ok(source) = fs::read_to_string(&file) else {
            continue;
        };

        // Only enforce this on migrated files that already declare typed descriptor errors.
        if !source.contains("BuiltinErrorDescriptor") {
            continue;
        }
        let runtime_source = pre_test_source(&source);

        assert!(
            !runtime_source.contains("const IDENT_"),
            "{} defines IDENT_* constant despite descriptor-backed errors; keep identifier/message/code in BuiltinErrorDescriptor rows only",
            file.display()
        );

        for line in runtime_source.lines() {
            let trimmed = line.trim();
            let is_error_const = trimmed.starts_with("const ")
                && trimmed.contains(": &str")
                && (trimmed.contains("_MESSAGE") || trimmed.contains("_CODE"));
            assert!(
                !is_error_const,
                "{} defines standalone *_MESSAGE/*_CODE constant in descriptor-migrated file; keep stable error fields in BuiltinErrorDescriptor rows",
                file.display()
            );
        }

        assert!(
            !runtime_source.contains("with_identifier(\"RunMat:"),
            "{} hard-codes a RunMat identifier via with_identifier(...); throw through BuiltinErrorDescriptor rows instead",
            file.display()
        );
    }
}
