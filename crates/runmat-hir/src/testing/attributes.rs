use crate::SemanticAttribute;

pub(super) fn has(attributes: &[SemanticAttribute], name: &str) -> bool {
    attributes
        .iter()
        .any(|attribute| attribute.name.eq_ignore_ascii_case(name))
}

pub(super) fn tags(attributes: &[SemanticAttribute]) -> Vec<String> {
    let mut tags = attributes
        .iter()
        .filter(|attribute| attribute.name.eq_ignore_ascii_case("TestTags"))
        .filter_map(|attribute| attribute.value.as_deref())
        .flat_map(quoted_values)
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    tags
}

fn quoted_values(raw: &str) -> Vec<String> {
    let mut values = Vec::new();
    let bytes = raw.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        let quote = bytes[index] as char;
        if quote != '\'' && quote != '"' {
            index += 1;
            continue;
        }
        index += 1;
        let mut value = String::new();
        while index < bytes.len() {
            let current = bytes[index] as char;
            if current == quote {
                if index + 1 < bytes.len() && bytes[index + 1] as char == quote {
                    value.push(quote);
                    index += 2;
                    continue;
                }
                index += 1;
                break;
            }
            value.push(current);
            index += 1;
        }
        if !value.is_empty() {
            values.push(value);
        }
    }
    if values.is_empty() {
        let bare = raw
            .trim()
            .trim_matches(['{', '}', '[', ']', '(', ')'])
            .trim();
        if !bare.is_empty() {
            values.push(bare.to_owned());
        }
    }
    values
}
