use super::super::*;
use super::text::{ShapeCopy, TextCopy};

pub(in crate::execution::value_codec::decode::machine) enum TextValueTask<'a> {
    String {
        text: TextCopy<'a>,
        path: String,
    },
    Character {
        shape: ShapeCopy<'a>,
        source: &'a [u32],
        chars: Vec<char>,
        path: String,
    },
    StringArray {
        shape: ShapeCopy<'a>,
        source: &'a [String],
        values: Vec<String>,
        current: Option<TextCopy<'a>>,
        path: String,
    },
}

impl<'a> TextValueTask<'a> {
    pub(super) fn string(source: &'a str, path: String) -> Self {
        Self::String {
            text: TextCopy::new(source),
            path,
        }
    }

    pub(super) fn character(shape: &'a [u64], source: &'a [u32], path: String) -> Self {
        Self::Character {
            shape: ShapeCopy::new(shape, path.clone()),
            source,
            chars: Vec::with_capacity(source.len()),
            path,
        }
    }

    pub(super) fn string_array(shape: &'a [u64], source: &'a [String], path: String) -> Self {
        Self::StringArray {
            shape: ShapeCopy::new(shape, path.clone()),
            source,
            values: Vec::with_capacity(source.len()),
            current: None,
            path,
        }
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        match self {
            Self::String { text, path: _ } => {
                let completed = text.advance(maximum_work);
                if !text.is_complete() {
                    return Ok((completed, None));
                }
                let text = std::mem::replace(text, TextCopy::new("")).finish();
                Ok((completed, Some(Value::String(text))))
            }
            Self::Character {
                shape,
                source,
                chars,
                path,
            } => {
                if !shape.is_complete() {
                    let completed = shape.advance(maximum_work)?;
                    return Ok((completed, None));
                }
                let count = (source.len() - chars.len()).min(maximum_work);
                for point in &source[chars.len()..chars.len() + count] {
                    chars.push(
                        char::from_u32(*point).ok_or_else(|| {
                            ValueCodecError::invalid(path, "invalid Unicode scalar")
                        })?,
                    );
                }
                if chars.len() != source.len() {
                    return Ok((count.max(1), None));
                }
                let shape = std::mem::replace(shape, ShapeCopy::new(&[], String::new())).finish();
                let chars = std::mem::take(chars);
                if shape.is_empty() {
                    let value = CharArray::new_with_shape(chars, shape)
                        .map(Value::CharArray)
                        .map_err(|error| ValueCodecError::invalid(path, error))?;
                    return Ok((count.max(1), Some(value)));
                }
                let (rows, cols) = rows_cols(&shape);
                let value = Value::CharArray(CharArray {
                    data: chars,
                    shape,
                    rows,
                    cols,
                });
                Ok((count.max(1), Some(value)))
            }
            Self::StringArray {
                shape,
                source,
                values,
                current,
                path: _,
            } => {
                if !shape.is_complete() {
                    let completed = shape.advance(maximum_work)?;
                    return Ok((completed, None));
                }
                if values.len() == source.len() {
                    let shape =
                        std::mem::replace(shape, ShapeCopy::new(&[], String::new())).finish();
                    let values = std::mem::take(values);
                    let (rows, cols) = rows_cols(&shape);
                    let value = Value::StringArray(StringArray {
                        data: values,
                        shape,
                        rows,
                        cols,
                    });
                    return Ok((1, Some(value)));
                }
                let copy = current.get_or_insert_with(|| TextCopy::new(&source[values.len()]));
                let completed = copy.advance(maximum_work);
                if copy.is_complete() {
                    values.push(current.take().expect("active text copy").finish());
                }
                Ok((completed, None))
            }
        }
    }

    pub(super) fn path(&self) -> &str {
        match self {
            Self::String { path, .. }
            | Self::Character { path, .. }
            | Self::StringArray { path, .. } => path,
        }
    }
}

fn rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [length] => (1, *length),
        [rows, columns, ..] => (*rows, *columns),
    }
}
