#[derive(Clone, Debug)]
pub struct RedactionPolicy {
    secrets: Vec<String>,
    pub max_text_bytes: usize,
}

impl RedactionPolicy {
    pub fn new(secrets: impl IntoIterator<Item = String>, max_text_bytes: usize) -> Self {
        let mut secrets: Vec<_> = secrets
            .into_iter()
            .filter(|value| !value.is_empty())
            .collect();
        secrets.sort_by_key(|value| std::cmp::Reverse(value.len()));
        secrets.dedup();
        Self {
            secrets,
            max_text_bytes,
        }
    }

    pub fn redact(&self, value: &str) -> RedactedText {
        self.redact_with_limit(value, self.max_text_bytes)
    }

    pub fn redact_with_limit(&self, value: &str, max_text_bytes: usize) -> RedactedText {
        let mut text = value.to_owned();
        for secret in &self.secrets {
            text = text.replace(secret, "[REDACTED]");
        }
        let truncated = text.len() > max_text_bytes;
        if truncated {
            let mut boundary = max_text_bytes;
            while boundary > 0 && !text.is_char_boundary(boundary) {
                boundary -= 1;
            }
            text.truncate(boundary);
        }
        RedactedText { text, truncated }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RedactedText {
    pub text: String,
    pub truncated: bool,
}
