use super::*;

pub(super) fn generated_variable_names(count: usize) -> Vec<String> {
    (1..=count).map(|idx| format!("Var{idx}")).collect()
}

pub(super) fn make_unique_variable_names(names: Vec<String>) -> Vec<String> {
    make_unique_names(
        names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| make_valid_variable_name(&name, idx + 1))
            .collect(),
    )
}

pub(super) fn make_unique_names(names: Vec<String>) -> Vec<String> {
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(names.len());
    for (idx, name) in names.into_iter().enumerate() {
        let base = if name.trim().is_empty() {
            format!("Var{}", idx + 1)
        } else {
            name.trim().to_string()
        };
        let mut candidate = base.clone();
        let mut suffix = 1usize;
        while used.contains(&candidate.to_ascii_lowercase()) {
            suffix += 1;
            candidate = format!("{base}_{suffix}");
        }
        used.insert(candidate.to_ascii_lowercase());
        out.push(candidate);
    }
    out
}

pub(super) fn make_valid_variable_name(raw: &str, fallback_index: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in raw.trim().chars().enumerate() {
        if (idx == 0 && (ch.is_ascii_alphabetic() || ch == '_'))
            || (idx > 0 && (ch.is_ascii_alphanumeric() || ch == '_'))
        {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() || !out.chars().next().unwrap().is_ascii_alphabetic() {
        format!("Var{fallback_index}")
    } else {
        out
    }
}
