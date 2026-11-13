//! Value-free regexp/regexpi option grammar and output projection.

use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexOutput {
    Start,
    End,
    Match,
    Tokens,
    TokenExtents,
    Names,
    Split,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyMatchPolicy {
    Remove,
    Allow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexOptions {
    pub outputs: Vec<RegexOutput>,
    pub once: bool,
    pub force_cell_output: bool,
    pub case_insensitive: bool,
    pub has_case_directive: bool,
    pub multi_line: bool,
    pub dot_all: bool,
    pub emptymatch: EmptyMatchPolicy,
}

impl Default for RegexOptions {
    fn default() -> Self {
        Self {
            outputs: Vec::new(),
            once: false,
            force_cell_output: false,
            case_insensitive: false,
            has_case_directive: false,
            multi_line: false,
            dot_all: true,
            emptymatch: EmptyMatchPolicy::Remove,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexOptionError {
    NonText(usize),
    Unknown { index: usize, name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexParseMode {
    Runtime,
    Inference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexOptionProjection {
    pub options: RegexOptions,
    /// False when an unobserved literal could change option consumption or outputs.
    pub complete: bool,
}

impl RegexOptions {
    pub fn outputs_for_multi(&self) -> Vec<RegexOutput> {
        if self.outputs.is_empty() {
            vec![
                RegexOutput::Start,
                RegexOutput::End,
                RegexOutput::TokenExtents,
                RegexOutput::Match,
                RegexOutput::Tokens,
                RegexOutput::Names,
                RegexOutput::Split,
            ]
        } else {
            self.outputs.clone()
        }
    }

    pub fn outputs_for_single(&self) -> Vec<RegexOutput> {
        if self.outputs.is_empty() {
            vec![RegexOutput::Start]
        } else {
            self.outputs.clone()
        }
    }

    pub fn apply_regexpi_default(&mut self) {
        if !self.has_case_directive {
            self.case_insensitive = true;
        }
    }
}

/// Requests only positions inspected by the grammar. `None` denotes an
/// unobserved or non-text operand; runtime rejects it on encounter, while
/// inference keeps the projection dynamic. Ignored warning operands are never
/// requested from the caller.
pub fn project_regex_options<'a>(
    len: usize,
    mut text_at: impl FnMut(usize) -> Option<Cow<'a, str>>,
    mode: RegexParseMode,
) -> Result<RegexOptionProjection, RegexOptionError> {
    let mut options = RegexOptions::default();
    let mut complete = true;
    let mut index = 0;
    while index < len {
        let Some(raw) = text_at(index) else {
            if mode == RegexParseMode::Runtime {
                return Err(RegexOptionError::NonText(index));
            }
            complete = false;
            index += 1;
            continue;
        };
        let name = raw.to_ascii_lowercase();
        index += 1;
        match name.as_str() {
            "start" => options.outputs.push(RegexOutput::Start),
            "end" => options.outputs.push(RegexOutput::End),
            "match" => options.outputs.push(RegexOutput::Match),
            "tokens" => options.outputs.push(RegexOutput::Tokens),
            "tokenextents" => options.outputs.push(RegexOutput::TokenExtents),
            "names" => options.outputs.push(RegexOutput::Names),
            "split" => options.outputs.push(RegexOutput::Split),
            "once" => options.once = true,
            "forcecelloutput" => options.force_cell_output = true,
            "ignorecase" => {
                options.case_insensitive = true;
                options.has_case_directive = true;
            }
            "matchcase" => {
                options.case_insensitive = false;
                options.has_case_directive = true;
            }
            "lineanchors" | "dotall" | "dotexceptnewline" => {
                let toggle = (index < len).then(|| text_at(index)).flatten();
                let flag = match toggle.as_deref() {
                    Some(value) if value.eq_ignore_ascii_case("on") => Some(true),
                    Some(value) if value.eq_ignore_ascii_case("off") => Some(false),
                    _ => None,
                };
                if flag.is_some() {
                    index += 1;
                } else if index < len && toggle.is_none() {
                    complete = false;
                }
                let flag = flag.unwrap_or(true);
                match name.as_str() {
                    "lineanchors" => options.multi_line = flag,
                    "dotall" => options.dot_all = flag,
                    _ => options.dot_all = !flag,
                }
            }
            "emptymatch" => {
                options.emptymatch = EmptyMatchPolicy::Allow;
                let next = (index < len).then(|| text_at(index)).flatten();
                match next.as_deref() {
                    Some(value) if value.eq_ignore_ascii_case("allow") => index += 1,
                    Some(value) if value.eq_ignore_ascii_case("remove") => {
                        options.emptymatch = EmptyMatchPolicy::Remove;
                        index += 1;
                    }
                    _ => {
                        if index < len && next.is_none() {
                            complete = false;
                        }
                    }
                }
            }
            "noemptymatch" => options.emptymatch = EmptyMatchPolicy::Remove,
            "stringanchors" => options.multi_line = false,
            "warnings" => {
                // Preserve the historical Runtime operand consumption, including
                // a following output selector or malformed value.
                if index < len {
                    index += 1;
                }
            }
            _ if mode == RegexParseMode::Runtime => {
                return Err(RegexOptionError::Unknown {
                    index: index - 1,
                    name,
                });
            }
            _ => complete = false,
        }
    }
    Ok(RegexOptionProjection { options, complete })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project_words(
        values: &[&str],
        mode: RegexParseMode,
    ) -> Result<RegexOptionProjection, RegexOptionError> {
        project_regex_options(
            values.len(),
            |index| Some(Cow::Borrowed(values[index])),
            mode,
        )
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn consumed_controls_do_not_become_selectors() {
        let options = project_words(
            &[
                "lineanchors",
                "off",
                "emptymatch",
                "remove",
                "warnings",
                "match",
                "split",
            ],
            RegexParseMode::Runtime,
        )
        .unwrap()
        .options;
        assert!(!options.multi_line);
        assert_eq!(options.emptymatch, EmptyMatchPolicy::Remove);
        assert_eq!(options.outputs, vec![RegexOutput::Split]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn warning_consumption_does_not_disable_regexpi_default() {
        let mut options = project_words(&["warnings", "matchcase"], RegexParseMode::Runtime)
            .unwrap()
            .options;
        options.apply_regexpi_default();
        assert!(options.case_insensitive);
        assert!(!options.has_case_directive);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn direct_projection_normalizes_optional_lookahead_operands() {
        let options = project_words(
            &["LINEANCHORS", "OFF", "EMPTYMATCH", "REMOVE", "MATCH"],
            RegexParseMode::Runtime,
        )
        .unwrap()
        .options;
        assert!(!options.multi_line);
        assert_eq!(options.emptymatch, EmptyMatchPolicy::Remove);
        assert_eq!(options.outputs, vec![RegexOutput::Match]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn lazy_reader_skips_warning_operand_and_stops_at_first_invalid_option() {
        let values = ["warnings", "unread", "match"];
        let mut accessed = Vec::new();
        let projection = project_regex_options(
            values.len(),
            |index| {
                accessed.push(index);
                Some(Cow::Borrowed(values[index]))
            },
            RegexParseMode::Runtime,
        )
        .unwrap();
        assert_eq!(accessed, vec![0, 2]);
        assert_eq!(projection.options.outputs, vec![RegexOutput::Match]);

        let values = ["invalid", "unread"];
        accessed.clear();
        let error = project_regex_options(
            values.len(),
            |index| {
                accessed.push(index);
                Some(Cow::Borrowed(values[index]))
            },
            RegexParseMode::Runtime,
        )
        .unwrap_err();
        assert_eq!(accessed, vec![0]);
        assert_eq!(
            error,
            RegexOptionError::Unknown {
                index: 0,
                name: "invalid".into()
            }
        );
    }
}
