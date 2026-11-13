use super::*;

#[derive(Debug, Clone)]
pub(super) enum ColumnData {
    Numeric {
        dtype: NumericDType,
        values: Vec<NumericScalar>,
    },
    Text(Vec<String>),
}

impl ColumnData {
    pub(super) fn new(kind: OutputKind) -> Self {
        match kind {
            OutputKind::Numeric(dtype) => ColumnData::Numeric {
                dtype,
                values: Vec::new(),
            },
            OutputKind::Text => ColumnData::Text(Vec::new()),
        }
    }

    pub(super) fn kind(&self) -> OutputKind {
        match self {
            ColumnData::Numeric { dtype, .. } => OutputKind::Numeric(*dtype),
            ColumnData::Text(_) => OutputKind::Text,
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            ColumnData::Numeric { values, .. } => values.len(),
            ColumnData::Text(values) => values.len(),
        }
    }

    pub(super) fn truncate(&mut self, len: usize) {
        match self {
            ColumnData::Numeric { values, .. } => values.truncate(len),
            ColumnData::Text(values) => values.truncate(len),
        }
    }

    pub(super) fn push_numeric(&mut self, value: NumericScalar) {
        let ColumnData::Numeric { dtype, values } = self else {
            unreachable!("numeric pushed into text column");
        };
        debug_assert_eq!(*dtype, value.numeric_dtype());
        values.push(value);
    }

    pub(super) fn push_text(&mut self, value: String) {
        let ColumnData::Text(values) = self else {
            unreachable!("text pushed into numeric column");
        };
        values.push(value);
    }
}
