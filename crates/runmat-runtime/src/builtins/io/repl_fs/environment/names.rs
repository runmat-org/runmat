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
        shape: Vec<usize>,
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
    Encoding(runmat_value::Utf16DecodeError),
}

impl EnvironmentNames {
    pub(super) fn decode(value: &Value) -> Result<Self, NameError> {
        match value {
            Value::CharArray(array)
                if crate::builtins::common::shape::effective_rank(&array.shape) > 2 =>
            {
                Err(NameError::InvalidType)
            }
            Value::CharArray(array) if array.rows == 1 => {
                Ok(Self::CharacterRow(character_row(array, 0)?))
            }
            Value::CharArray(array) if array.rows > 1 => Ok(Self::CharacterMatrix {
                names: (0..array.rows)
                    .map(|row| character_row(array, row))
                    .collect::<Result<Vec<_>, _>>()?,
                rows: array.rows,
            }),
            Value::CharArray(_) => Ok(Self::CharacterRow(String::new())),
            Value::String(runmat_value::StringElement::Text(value)) => {
                Ok(Self::StringScalar(value.as_str().to_owned()))
            }
            Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
                Ok(Self::Strings {
                    names: array
                        .data()
                        .iter()
                        .map(|text| text.as_str().expect("checked nonmissing string").to_owned())
                        .collect(),
                    shape: array.shape.as_slice().to_vec(),
                })
            }
            Value::Cell(array) => Self::decode_cells(array),
            _ => Err(NameError::InvalidType),
        }
    }

    fn decode_cells(array: &CellArray) -> Result<Self, NameError> {
        let mut names = Vec::with_capacity(array.data().len());
        let mut kinds = Vec::with_capacity(array.data().len());
        for value in array.iter_column_major() {
            match value {
                Value::CharArray(chars)
                    if chars.rows == 1
                        && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
                {
                    names.push(character_row(chars, 0)?);
                    kinds.push(CellNameKind::Character);
                }
                Value::String(runmat_value::StringElement::Text(text)) => {
                    names.push(text.as_str().to_owned());
                    kinds.push(CellNameKind::String);
                }
                _ => return Err(NameError::InvalidCellElement),
            }
        }
        Ok(Self::Cells {
            names,
            kinds,
            shape: array.shape.as_slice().to_vec(),
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
            Self::Cells { shape, .. } => Some(shape.clone()),
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

fn character_row(array: &CharArray, row: usize) -> Result<String, NameError> {
    let start = row * array.cols;
    let mut text = runmat_value::decode_utf16_code_units(&array.data()[start..start + array.cols])
        .map_err(NameError::Encoding)?;
    while text.ends_with(' ') {
        text.pop();
    }
    Ok(text)
}
