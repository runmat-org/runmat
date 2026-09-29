use super::*;

#[derive(Debug)]
pub(super) struct ExportOptions {
    pub(super) opset_version: i64,
    pub(super) batch_size: Option<i64>,
    pub(super) input_name: Option<String>,
    pub(super) output_name: Option<String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            opset_version: DEFAULT_OPSET_VERSION,
            batch_size: None,
            input_name: None,
            output_name: None,
        }
    }
}

#[derive(Debug)]
pub(super) struct ExportGraph {
    pub(super) nodes: Vec<Node>,
    pub(super) initializers: Vec<Initializer>,
    pub(super) input: ValueInfo,
    pub(super) output: ValueInfo,
    pub(super) dtype: NumericDType,
}

#[derive(Debug)]
pub(super) struct Node {
    pub(super) name: String,
    pub(super) op_type: String,
    pub(super) inputs: Vec<String>,
    pub(super) outputs: Vec<String>,
    pub(super) attributes: Vec<Attribute>,
}

#[derive(Debug)]
pub(super) struct Initializer {
    pub(super) name: String,
    pub(super) dims: Vec<i64>,
    pub(super) data: NumericStorage,
}

#[derive(Debug)]
pub(super) struct ValueInfo {
    pub(super) name: String,
    pub(super) batch_size: Option<i64>,
    pub(super) features: i64,
}

#[derive(Debug)]
pub(super) enum Attribute {
    Float { name: &'static str, value: f32 },
    Int { name: &'static str, value: i64 },
}
