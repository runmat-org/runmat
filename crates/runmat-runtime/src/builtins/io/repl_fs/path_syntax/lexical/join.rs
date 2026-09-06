pub(in crate::builtins::io::repl_fs::path_syntax) fn join(parts: &[&str]) -> String {
    let separator = std::path::MAIN_SEPARATOR;
    let mut combined = String::new();
    for part in parts.iter().filter(|part| !part.is_empty()) {
        let normalized = platform_separators(part);
        if combined.is_empty() {
            combined.push_str(&normalized);
            continue;
        }
        if !combined.ends_with(separator) {
            combined.push(separator);
        }
        combined.push_str(normalized.trim_start_matches(separator));
    }
    normalize_joined(&combined)
}

fn normalize_joined(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let separator = std::path::MAIN_SEPARATOR;
    let leading_separator_count = path
        .chars()
        .take_while(|character| *character == separator)
        .count();
    let trailing_separator = path.ends_with(separator);
    let raw_components: Vec<&str> = path
        .trim_start_matches(separator)
        .split(separator)
        .filter(|component| !component.is_empty())
        .collect();
    let mut components = Vec::with_capacity(raw_components.len());
    for (index, component) in raw_components.iter().enumerate() {
        let is_terminal_dot =
            *component == "." && index + 1 == raw_components.len() && !trailing_separator;
        if *component != "." || is_terminal_dot {
            components.push(*component);
        }
    }
    let mut normalized = separator.to_string().repeat(leading_separator_count);
    for component in components {
        if !normalized.is_empty() && !normalized.ends_with(separator) {
            normalized.push(separator);
        }
        normalized.push_str(component);
    }
    if trailing_separator && !normalized.is_empty() && !normalized.ends_with(separator) {
        normalized.push(separator);
    }
    normalized
}

fn platform_separators(text: &str) -> String {
    if cfg!(windows) {
        text.replace('/', "\\")
    } else {
        text.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_with_the_platform_separator() {
        let separator = std::path::MAIN_SEPARATOR;
        assert_eq!(join(&["one", "two"]), format!("one{separator}two"));
    }

    #[test]
    fn collapses_only_inner_separators_and_nonterminal_dots() {
        let separator = std::path::MAIN_SEPARATOR;
        let repeated = separator.to_string().repeat(2);
        assert_eq!(
            join(&["root", &format!("{repeated}folder{repeated}")]),
            format!("root{separator}folder{separator}")
        );
        assert_eq!(
            join(&[
                "root",
                &format!(".{separator}child"),
                &format!("..{separator}leaf{separator}.")
            ]),
            format!("root{separator}child{separator}..{separator}leaf{separator}.")
        );
        assert_eq!(
            join(&["root", &format!("{separator}child")]),
            format!("root{separator}child")
        );
        assert!(join(&[&format!("{repeated}server"), "share"]).starts_with(&repeated));
    }
}
