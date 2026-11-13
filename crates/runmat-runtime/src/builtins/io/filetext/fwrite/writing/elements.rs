use super::*;
use std::str::Chars;

#[derive(Clone, Debug)]
pub(in super::super) enum WriteElement {
    Floating(f64),
    Integer(IntValue),
}

impl WriteElement {
    pub(in super::super) fn as_f64(&self) -> f64 {
        match self {
            Self::Floating(value) => *value,
            Self::Integer(value) => value.to_f64(),
        }
    }
}

pub(in super::super) enum BorrowedElements<'a> {
    Tensor {
        tensor: &'a runmat_value::Tensor,
        index: usize,
    },
    Scalar(Option<WriteElement>),
    Logical(std::slice::Iter<'a, u8>),
    Characters {
        array: &'a CharArray,
        index: usize,
        page_len: usize,
    },
    String(Chars<'a>),
    Strings {
        array: &'a runmat_value::StringArray,
        index: usize,
        chars: Option<Chars<'a>>,
    },
}

pub(in super::super) fn borrow_elements(value: &Value) -> Result<BorrowedElements<'_>, String> {
    match value {
        Value::Tensor(tensor) => Ok(BorrowedElements::Tensor { tensor, index: 0 }),
        Value::Num(n) => Ok(BorrowedElements::Scalar(Some(WriteElement::Floating(*n)))),
        Value::Int(int) => Ok(BorrowedElements::Scalar(Some(WriteElement::Integer(
            int.clone(),
        )))),
        Value::Bool(b) => Ok(BorrowedElements::Scalar(Some(WriteElement::Floating(
            if *b { 1.0 } else { 0.0 },
        )))),
        Value::LogicalArray(array) => Ok(BorrowedElements::Logical(array.data.iter())),
        Value::CharArray(array) => Ok(BorrowedElements::Characters {
            array,
            index: 0,
            page_len: array.rows.checked_mul(array.cols).unwrap_or(0),
        }),
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(BorrowedElements::String(text.chars()))
        }
        Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
            Ok(BorrowedElements::Strings {
                array,
                index: 0,
                chars: None,
            })
        }
        Value::GpuTensor(_) => Err("fwrite: expected host tensor data after gathering".to_string()),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("fwrite: complex values are not supported yet".to_string())
        }
        _ => Err(format!("fwrite: unsupported data type {:?}", value)),
    }
}

impl<'a> Iterator for BorrowedElements<'a> {
    type Item = WriteElement;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Tensor { tensor, index } => {
                if *index >= tensor.len() {
                    return None;
                }
                let value = tensor
                    .numeric_value_at(*index)
                    .expect("index within authoritative numeric storage");
                *index += 1;
                Some(value.into_int_value().map_or_else(
                    || WriteElement::Floating(value.materialize_f64()),
                    WriteElement::Integer,
                ))
            }
            Self::Scalar(value) => value.take(),
            Self::Logical(bits) => bits
                .next()
                .map(|bit| WriteElement::Floating(if *bit != 0 { 1.0 } else { 0.0 })),
            Self::Characters {
                array,
                index,
                page_len,
            } => {
                if *page_len == 0 || *index >= array.data().len() {
                    return None;
                }
                let page = *index / *page_len;
                let position = *index % *page_len;
                let row = position % array.rows;
                let column = position / array.rows;
                let source = page * *page_len + row * array.cols + column;
                *index += 1;
                array
                    .data()
                    .get(source)
                    .map(|ch| WriteElement::Floating(ch.raw() as f64))
            }
            Self::String(chars) => chars
                .next()
                .map(|ch| WriteElement::Floating(ch as u32 as f64)),
            Self::Strings {
                array,
                index,
                chars,
            } => loop {
                if let Some(active) = chars.as_mut() {
                    if let Some(ch) = active.next() {
                        return Some(WriteElement::Floating(ch as u32 as f64));
                    }
                    *chars = None;
                }
                let owner: &'a runmat_value::StringArray = array;
                let text: &'a str = owner
                    .data()
                    .get(*index)?
                    .as_str()
                    .expect("checked nonmissing string data");
                let separate = *index > 0;
                *index += 1;
                *chars = Some(text.chars());
                if separate {
                    return Some(WriteElement::Floating('\n' as u32 as f64));
                }
            },
        }
    }
}
