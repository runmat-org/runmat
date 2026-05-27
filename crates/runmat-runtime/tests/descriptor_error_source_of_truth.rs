use std::collections::HashMap;
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

fn is_upper_snake_token(token: &str) -> bool {
    let token = token.trim();
    !token.is_empty()
        && token
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn parse_inline_string_literal(trimmed_line: &str) -> Option<String> {
    let first_quote = trimmed_line.find('"')?;
    let remainder = &trimmed_line[first_quote + 1..];
    let second_quote = remainder.find('"')?;
    Some(remainder[..second_quote].to_string())
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
        let mut descriptor_identifier_counts: HashMap<String, usize> = HashMap::new();
        let mut descriptor_code_counts: HashMap<String, usize> = HashMap::new();
        let mut descriptor_message_counts: HashMap<String, usize> = HashMap::new();

        assert!(
            !source.contains("const IDENT_"),
            "{} defines IDENT_* constant despite descriptor-backed errors; keep identifier/message/code in BuiltinErrorDescriptor rows only",
            file.display()
        );

        let mut in_error_descriptor = false;
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

            if trimmed.starts_with("const ")
                && trimmed.contains(": BuiltinErrorDescriptor = BuiltinErrorDescriptor {")
            {
                in_error_descriptor = true;
                continue;
            }
            if in_error_descriptor && trimmed == "};" {
                in_error_descriptor = false;
                continue;
            }
            if !in_error_descriptor {
                continue;
            }

            if trimmed.starts_with("identifier:") && trimmed.contains("Some(") {
                assert!(
                    trimmed.contains("Some(\""),
                    "{} forwards identifier via constant/expression ({trimmed}); keep identifier text authored inline in BuiltinErrorDescriptor rows",
                    file.display()
                );
                if let Some(identifier) = parse_inline_string_literal(trimmed) {
                    *descriptor_identifier_counts.entry(identifier).or_insert(0) += 1;
                }
            }

            if trimmed.starts_with("code:") {
                let rhs = trimmed.trim_start_matches("code:").trim_start();
                let token = rhs.trim_end_matches(',').trim();
                if is_upper_snake_token(token) {
                    panic!(
                        "{} forwards error code via constant/expression ({trimmed}); keep code text authored inline in BuiltinErrorDescriptor rows",
                        file.display()
                    );
                }
                if let Some(code) = parse_inline_string_literal(trimmed) {
                    *descriptor_code_counts.entry(code).or_insert(0) += 1;
                }
            }

            if trimmed.starts_with("message:") {
                let rhs = trimmed.trim_start_matches("message:").trim_start();
                let token = rhs.trim_end_matches(',').trim();
                if is_upper_snake_token(token) {
                    panic!(
                        "{} forwards error message via constant/expression ({trimmed}); keep message text authored inline in BuiltinErrorDescriptor rows",
                        file.display()
                    );
                }
                if let Some(message) = parse_inline_string_literal(trimmed) {
                    *descriptor_message_counts.entry(message).or_insert(0) += 1;
                }
            }
        }

        assert!(
            !runtime_source.contains("with_identifier(\"RunMat:"),
            "{} hard-codes a RunMat identifier via with_identifier(...); throw through BuiltinErrorDescriptor rows instead",
            file.display()
        );

        for (identifier, expected_count) in descriptor_identifier_counts {
            let quoted_identifier = format!("\"{identifier}\"");
            let actual_count = runtime_source.matches(&quoted_identifier).count();
            assert_eq!(
                actual_count, expected_count,
                "{} duplicates stable identifier literal `{}` outside BuiltinErrorDescriptor rows; keep identifier authored only in descriptor rows",
                file.display(), identifier
            );
        }

        for (code, expected_count) in descriptor_code_counts {
            let quoted_code = format!("\"{code}\"");
            let actual_count = runtime_source.matches(&quoted_code).count();
            assert_eq!(
                actual_count, expected_count,
                "{} duplicates stable error code literal `{}` outside BuiltinErrorDescriptor rows; keep code authored only in descriptor rows",
                file.display(), code
            );
        }

        for (message, expected_count) in descriptor_message_counts {
            let quoted_message = format!("\"{message}\"");
            let actual_count = runtime_source.matches(&quoted_message).count();
            assert_eq!(
                actual_count, expected_count,
                "{} duplicates stable error message literal `{}` outside BuiltinErrorDescriptor rows; keep message authored only in descriptor rows",
                file.display(), message
            );
        }
    }
}
