mod coverage;
mod fanout;

pub use coverage::CoveragePlugin;
pub use fanout::{PluginFanout, TestPlugin};

use crate::reporter::RenderedReport;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginOutput {
    pub message: Option<String>,
    pub reports: Vec<RenderedReport>,
}

impl PluginOutput {
    pub fn message(message: impl Into<String>) -> Self {
        Self {
            message: Some(message.into()),
            reports: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginError {
    pub message: String,
}

impl PluginError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
