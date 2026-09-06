use runmat_value::{CellArray, CharArray, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NameFeature {
    CharacterMatrix,
    StringInCell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CellNameKind {
    Character,
    String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum EnvironmentNames {
    CharacterRow(String),
    StringScalar(String),
    Strings {
        names: Vec<String>,
        shape: Vec<usize>,
    },
    Cells {
        names: Vec<String>,
        kinds: Vec<CellNameKind>,
        rows: usize,
        cols: usize,
    },
    CharacterMatrix {
        names: Vec<String>,
        rows: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NameError {
    InvalidType,
    InvalidCellElement,
}

impl EnvironmentNames {
    pub(super) fn decode(value: &Value) -> Result<Self, NameError> {
        match value {
            Value::CharArray(array) if array.rows == 1 => {
                Ok(Self::CharacterRow(character_row(array, 0)))
            }
            Value::CharArray(array) if array.rows > 1 => Ok(Self::CharacterMatrix {
                names: (0..array.rows)
                    .map(|row| character_row(array, row))
                    .collect(),
                rows: array.rows,
            }),
            Value::CharArray(_) => Ok(Self::CharacterRow(String::new())),
            Value::String(value) => Ok(Self::StringScalar(value.clone())),
            Value::StringArray(array) => Ok(Self::Strings {
                names: array.data.clone(),
                shape: array.shape.clone(),
            }),
            Value::Cell(array) => Self::decode_cells(array),
            _ => Err(NameError::InvalidType),
        }
    }

    fn decode_cells(array: &CellArray) -> Result<Self, NameError> {
        let mut names = Vec::with_capacity(array.data.len());
        let mut kinds = Vec::with_capacity(array.data.len());
        for value in &array.data {
            match value {
                Value::CharArray(chars) if chars.rows == 1 => {
                    names.push(character_row(chars, 0));
                    kinds.push(CellNameKind::Character);
                }
                Value::String(text) => {
                    names.push(text.clone());
                    kinds.push(CellNameKind::String);
                }
                _ => return Err(NameError::InvalidCellElement),
            }
        }
        Ok(Self::Cells {
            names,
            kinds,
            rows: array.rows,
            cols: array.cols,
        })
    }

    pub(super) fn names(&self) -> &[String] {
        match self {
            Self::CharacterRow(name) | Self::StringScalar(name) => std::slice::from_ref(name),
            Self::Strings { names, .. }
            | Self::Cells { names, .. }
            | Self::CharacterMatrix { names, .. } => names,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.names().len()
    }

    pub(super) fn shape(&self) -> Option<Vec<usize>> {
        match self {
            Self::CharacterRow(_) | Self::StringScalar(_) => None,
            Self::Strings { shape, .. } => Some(shape.clone()),
            Self::Cells { rows, cols, .. } => Some(vec![*rows, *cols]),
            Self::CharacterMatrix { rows, .. } => Some(vec![*rows, 1]),
        }
    }

    pub(super) fn features(&self) -> impl Iterator<Item = NameFeature> + '_ {
        let character_matrix =
            matches!(self, Self::CharacterMatrix { .. }).then_some(NameFeature::CharacterMatrix);
        let string_in_cell = match self {
            Self::Cells { kinds, .. } if kinds.contains(&CellNameKind::String) => {
                Some(NameFeature::StringInCell)
            }
            _ => None,
        };
        character_matrix.into_iter().chain(string_in_cell)
    }
}

fn character_row(array: &CharArray, row: usize) -> String {
    let start = row * array.cols;
    let mut text: String = array.data[start..start + array.cols].iter().collect();
    while text.ends_with(' ') {
        text.pop();
    }
    text
}
