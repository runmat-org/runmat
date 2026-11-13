#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WinqueryregSelection {
    DefaultValue,
    NamedValue(String),
    ValueNames,
}

impl WinqueryregSelection {
    /// An absent or empty value-name selects the default value, never all values.
    pub fn value(name: Option<String>) -> Self {
        match name.filter(|name| !name.is_empty()) {
            Some(name) => Self::NamedValue(name),
            None => Self::DefaultValue,
        }
    }
}
