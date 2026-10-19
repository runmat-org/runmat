use serde::{Deserialize, Serialize, Serializer};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFieldRef {
    pub backend: String,
    pub token: String,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFieldValues {
    HostF64(Vec<f64>),
    DeviceRef(DeviceFieldRef),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnalysisField {
    pub field_id: String,
    pub shape: Vec<usize>,
    pub values: AnalysisFieldValues,
}

// One wire authority for owned fields and borrowed/hypothetical host sequences.
#[derive(Serialize)]
#[serde(rename = "AnalysisFieldValues", rename_all = "snake_case")]
enum BorrowedValues<'a, T: Serialize + ?Sized> {
    HostF64(&'a T),
    DeviceRef(&'a DeviceFieldRef),
}

#[derive(Serialize)]
#[serde(rename = "AnalysisField")]
struct BorrowedField<'a, T: Serialize + ?Sized> {
    field_id: &'a str,
    shape: &'a [usize],
    values: BorrowedValues<'a, T>,
}

impl Serialize for AnalysisFieldValues {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::HostF64(values) => {
                BorrowedValues::HostF64(values.as_slice()).serialize(serializer)
            }
            Self::DeviceRef(reference) => {
                BorrowedValues::<[f64]>::DeviceRef(reference).serialize(serializer)
            }
        }
    }
}

impl Serialize for AnalysisField {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let values = match &self.values {
            AnalysisFieldValues::HostF64(values) => BorrowedValues::HostF64(values.as_slice()),
            AnalysisFieldValues::DeviceRef(reference) => BorrowedValues::DeviceRef(reference),
        };
        BorrowedField {
            field_id: &self.field_id,
            shape: &self.shape,
            values,
        }
        .serialize(serializer)
    }
}

impl AnalysisField {
    /// Borrow the canonical host-field wire schema with caller-supplied serializable values.
    /// `values` must serialize as an f64 sequence. This is a wire/profile view, not validated
    /// numerical storage; it neither checks shape agreement nor materializes numeric values.
    pub fn host_f64_serialization<'a, T: Serialize + ?Sized>(
        field_id: &'a str,
        shape: &'a [usize],
        values: &'a T,
    ) -> impl Serialize + 'a {
        BorrowedField {
            field_id,
            shape,
            values: BorrowedValues::HostF64(values),
        }
    }
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

#[cfg(test)]
mod tests;
