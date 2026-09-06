use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::warning_store::RuntimeWarning;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WarningMode {
    On,
    Off,
    Once,
    Error,
}

impl WarningMode {
    pub(crate) fn keyword(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
            Self::Once => "once",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WarningState {
    pub(crate) identifier: String,
    pub(crate) mode: WarningMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WarningAction {
    Suppress,
    Display,
    AsError,
}

#[derive(Clone, Copy, Debug)]
struct WarningRule {
    mode: WarningMode,
    triggered: bool,
}

impl WarningRule {
    fn new(mode: WarningMode) -> Self {
        Self {
            mode,
            triggered: false,
        }
    }
}

#[derive(Debug)]
pub(crate) struct WarningPolicy {
    default_mode: WarningMode,
    rules: HashMap<String, WarningRule>,
    once_seen_default: HashSet<String>,
    backtrace_enabled: bool,
    verbose_enabled: bool,
    last_warning: Option<RuntimeWarning>,
}

impl Default for WarningPolicy {
    fn default() -> Self {
        Self {
            default_mode: WarningMode::On,
            rules: HashMap::new(),
            once_seen_default: HashSet::new(),
            backtrace_enabled: false,
            verbose_enabled: false,
            last_warning: None,
        }
    }
}

impl WarningPolicy {
    pub(crate) fn set_global_mode(&mut self, mode: WarningMode) -> WarningMode {
        let previous = self.default_mode;
        self.once_seen_default.clear();
        self.default_mode = mode;
        previous
    }

    pub(crate) fn set_identifier_mode(
        &mut self,
        identifier: &str,
        mode: WarningMode,
    ) -> WarningMode {
        let previous = self.lookup_mode(identifier);
        if mode == self.default_mode && !matches!(mode, WarningMode::Once) {
            self.rules.remove(identifier);
        } else {
            self.rules
                .insert(identifier.to_string(), WarningRule::new(mode));
        }
        if matches!(mode, WarningMode::Once) {
            self.once_seen_default.remove(identifier);
        }
        previous
    }

    pub(crate) fn clear_identifier(&mut self, identifier: &str) -> WarningMode {
        let previous = self.lookup_mode(identifier);
        self.rules.remove(identifier);
        self.once_seen_default.remove(identifier);
        previous
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn reset_defaults(&mut self) -> Vec<WarningState> {
        let snapshot = self.snapshot();
        self.default_mode = WarningMode::On;
        self.once_seen_default.clear();
        self.rules.clear();
        self.backtrace_enabled = false;
        self.verbose_enabled = false;
        snapshot
    }

    pub(crate) fn lookup_mode(&self, identifier: &str) -> WarningMode {
        self.rules
            .get(identifier)
            .map(|rule| rule.mode)
            .unwrap_or(self.default_mode)
    }

    pub(crate) fn set_backtrace(&mut self, enabled: bool) -> bool {
        std::mem::replace(&mut self.backtrace_enabled, enabled)
    }

    pub(crate) fn set_verbose(&mut self, enabled: bool) -> bool {
        std::mem::replace(&mut self.verbose_enabled, enabled)
    }

    pub(crate) fn backtrace_enabled(&self) -> bool {
        self.backtrace_enabled
    }

    pub(crate) fn verbose_enabled(&self) -> bool {
        self.verbose_enabled
    }

    pub(crate) fn last_warning(&self) -> Option<RuntimeWarning> {
        self.last_warning.clone()
    }

    #[cfg(test)]
    pub(crate) fn has_identifier_rules(&self) -> bool {
        !self.rules.is_empty()
    }

    pub(super) fn action_for(&mut self, identifier: &str) -> WarningAction {
        if let Some(rule) = self.rules.get_mut(identifier) {
            return match rule.mode {
                WarningMode::On => WarningAction::Display,
                WarningMode::Off => WarningAction::Suppress,
                WarningMode::Error => WarningAction::AsError,
                WarningMode::Once if rule.triggered => WarningAction::Suppress,
                WarningMode::Once => {
                    rule.triggered = true;
                    WarningAction::Display
                }
            };
        }
        match self.default_mode {
            WarningMode::On => WarningAction::Display,
            WarningMode::Off => WarningAction::Suppress,
            WarningMode::Error => WarningAction::AsError,
            WarningMode::Once if self.once_seen_default.contains(identifier) => {
                WarningAction::Suppress
            }
            WarningMode::Once => {
                self.once_seen_default.insert(identifier.to_string());
                WarningAction::Display
            }
        }
    }

    pub(super) fn record_last(&mut self, warning: RuntimeWarning) {
        self.last_warning = Some(warning);
    }

    pub(crate) fn snapshot(&self) -> Vec<WarningState> {
        let mut states = Vec::with_capacity(self.rules.len() + 3);
        states.push(WarningState {
            identifier: "all".to_string(),
            mode: self.default_mode,
        });
        let mut rules = self
            .rules
            .iter()
            .map(|(identifier, rule)| WarningState {
                identifier: identifier.clone(),
                mode: rule.mode,
            })
            .collect::<Vec<_>>();
        rules.sort_by(|left, right| left.identifier.cmp(&right.identifier));
        states.extend(rules);
        states.push(WarningState {
            identifier: "backtrace".to_string(),
            mode: if self.backtrace_enabled {
                WarningMode::On
            } else {
                WarningMode::Off
            },
        });
        states.push(WarningState {
            identifier: "verbose".to_string(),
            mode: if self.verbose_enabled {
                WarningMode::On
            } else {
                WarningMode::Off
            },
        });
        states
    }
}

thread_local! {
    static FALLBACK_POLICY: RefCell<WarningPolicy> = RefCell::new(WarningPolicy::default());
}

pub(crate) fn with_policy<R>(apply: impl FnOnce(&mut WarningPolicy) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        return apply(&mut context.state().warning_policy.borrow_mut());
    }
    FALLBACK_POLICY.with(|policy| apply(&mut policy.borrow_mut()))
}
