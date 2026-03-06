use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFieldRef {
    pub backend: String,
    pub token: String,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFieldValues {
    HostF64(Vec<f64>),
    DeviceRef(DeviceFieldRef),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisField {
    pub field_id: String,
    pub shape: Vec<usize>,
    pub values: AnalysisFieldValues,
}

impl AnalysisField {
    pub fn host_f64(field_id: impl Into<String>, shape: Vec<usize>, values: Vec<f64>) -> Self {
        Self {
            field_id: field_id.into(),
            shape,
            values: AnalysisFieldValues::HostF64(values),
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.element_count() == 0
    }

    pub fn as_host_f64(&self) -> Option<&[f64]> {
        match &self.values {
            AnalysisFieldValues::HostF64(values) => Some(values.as_slice()),
            AnalysisFieldValues::DeviceRef(_) => None,
        }
    }
}
