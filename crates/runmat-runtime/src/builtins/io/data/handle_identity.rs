//! Exact, bounded borrowed control identity for dataset admission.
//!
//! This is not a resource certificate. Values outside the synchronous work
//! budget are unavailable until a cooperative staged identity producer exists.
//! Fixed-width numeric encodings are bounded by the node limit; the byte
//! limit covers variable text and shape payloads, not every hash framing byte.

use runmat_execution::Digest;
use runmat_value::{NumericDType, NumericScalar, Value};
use sha2::{Digest as _, Sha256};

const MAX_NODES: usize = 4_096;
const MAX_TEXT_SHAPE_BYTES: usize = 1_048_576;
const MAX_DEPTH: usize = 64;

#[repr(u8)]
enum ValueTag {
    Text = 1,
    Character,
    Logical,
    Integer,
    Double,
    ComplexDouble,
    Tensor,
    ComplexTensor,
    LogicalArray,
    StringArray,
    Cell,
    Struct,
}

pub(super) struct HashBudget {
    nodes: usize,
    text_shape_bytes: usize,
    exhausted: bool,
}

impl HashBudget {
    pub(super) fn new() -> Self {
        Self {
            nodes: MAX_NODES,
            text_shape_bytes: MAX_TEXT_SHAPE_BYTES,
            exhausted: false,
        }
    }

    pub(super) fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    fn node(&mut self) -> Option<()> {
        self.nodes = self.nodes.checked_sub(1).or_else(|| {
            self.exhausted = true;
            None
        })?;
        Some(())
    }

    fn bytes(&mut self, count: usize) -> Option<()> {
        self.text_shape_bytes = self.text_shape_bytes.checked_sub(count).or_else(|| {
            self.exhausted = true;
            None
        })?;
        Some(())
    }
}

pub(super) fn exact_digest_with(budget: &mut HashBudget, value: &Value) -> Option<Digest> {
    if budget.exhausted {
        return None;
    }
    let mut hasher = Sha256::new();
    if hash_value(&mut hasher, budget, value, 0).is_none() {
        budget.exhausted = true;
        return None;
    }
    Some(Digest::from_bytes(hasher.finalize().into()))
}

fn hash_value(
    hash: &mut Sha256,
    budget: &mut HashBudget,
    value: &Value,
    depth: usize,
) -> Option<()> {
    if depth > MAX_DEPTH {
        return None;
    }
    budget.node()?;
    match value {
        Value::String(text) => {
            tag(hash, ValueTag::Text);
            string_element(hash, budget, text)?;
        }
        Value::CharArray(chars) => {
            tag(hash, ValueTag::Character);
            shape(hash, budget, &chars.shape)?;
            budget.nodes = budget.nodes.checked_sub(chars.data().len())?;
            budget.bytes(chars.data().len().checked_mul(2)?)?;
            for character in chars.data() {
                hash.update(character.raw().to_le_bytes());
            }
        }
        Value::Bool(boolean) => {
            tag(hash, ValueTag::Logical);
            hash.update([u8::from(*boolean)]);
        }
        Value::Int(integer) => {
            tag(hash, ValueTag::Integer);
            numeric(hash, budget, NumericScalar::from(integer.clone()))?;
        }
        Value::Num(number) => {
            tag(hash, ValueTag::Double);
            hash.update(number.to_bits().to_le_bytes());
        }
        Value::Complex(real, imaginary) => {
            tag(hash, ValueTag::ComplexDouble);
            hash.update(real.to_bits().to_le_bytes());
            hash.update(imaginary.to_bits().to_le_bytes());
        }
        Value::Tensor(tensor) => {
            tag(hash, ValueTag::Tensor);
            shape(hash, budget, tensor.shape())?;
            dtype(hash, tensor.numeric_dtype());
            for index in 0..tensor.len() {
                numeric(hash, budget, tensor.numeric_value_at(index)?)?;
            }
        }
        Value::ComplexTensor(tensor) => {
            tag(hash, ValueTag::ComplexTensor);
            shape(hash, budget, tensor.shape())?;
            dtype(hash, tensor.numeric_dtype());
            for index in 0..tensor.len() {
                let (real, imaginary) = tensor.numeric_value_at(index)?;
                numeric(hash, budget, real)?;
                numeric(hash, budget, imaginary)?;
            }
        }
        Value::LogicalArray(array) => {
            tag(hash, ValueTag::LogicalArray);
            shape(hash, budget, &array.shape)?;
            budget.bytes(array.data.len())?;
            hash.update(array.data.as_slice());
        }
        Value::StringArray(array) => {
            tag(hash, ValueTag::StringArray);
            shape(hash, budget, &array.shape)?;
            for text in array.data() {
                budget.node()?;
                string_element(hash, budget, text)?;
            }
        }
        Value::Cell(cell) => {
            tag(hash, ValueTag::Cell);
            shape(hash, budget, &cell.shape)?;
            for child in cell.data() {
                hash_value(hash, budget, child, depth + 1)?;
            }
        }
        Value::Struct(structure) => {
            tag(hash, ValueTag::Struct);
            length(hash, structure.fields.len());
            for (name, child) in &structure.fields {
                text_bytes(hash, budget, name)?;
                hash_value(hash, budget, child, depth + 1)?;
            }
        }
        _ => return None,
    }
    Some(())
}

fn tag(hash: &mut Sha256, value: ValueTag) {
    hash.update([value as u8]);
}

fn length(hash: &mut Sha256, value: usize) {
    hash.update((value as u64).to_le_bytes());
}

fn string_element(
    hash: &mut Sha256,
    budget: &mut HashBudget,
    value: &runmat_value::StringElement,
) -> Option<()> {
    match value.as_str() {
        Some(text) => {
            hash.update([1]);
            text_bytes(hash, budget, text)
        }
        None => {
            hash.update([0]);
            Some(())
        }
    }
}

fn text_bytes(hash: &mut Sha256, budget: &mut HashBudget, value: &str) -> Option<()> {
    budget.bytes(value.len())?;
    length(hash, value.len());
    hash.update(value.as_bytes());
    Some(())
}

fn shape(hash: &mut Sha256, budget: &mut HashBudget, dimensions: &[usize]) -> Option<()> {
    budget.nodes = budget.nodes.checked_sub(dimensions.len()).or_else(|| {
        budget.exhausted = true;
        None
    })?;
    budget.bytes(dimensions.len().checked_mul(8)?)?;
    length(hash, dimensions.len());
    for dimension in dimensions {
        length(hash, *dimension);
    }
    Some(())
}

fn dtype(hash: &mut Sha256, value: NumericDType) {
    let code = match value {
        NumericDType::F64 => 1,
        NumericDType::F32 => 2,
        NumericDType::I8 => 3,
        NumericDType::I16 => 4,
        NumericDType::I32 => 5,
        NumericDType::I64 => 6,
        NumericDType::U8 => 7,
        NumericDType::U16 => 8,
        NumericDType::U32 => 9,
        NumericDType::U64 => 10,
    };
    hash.update([code]);
}

fn numeric(hash: &mut Sha256, budget: &mut HashBudget, value: NumericScalar) -> Option<()> {
    budget.node()?;
    match value {
        NumericScalar::F64(value) => {
            hash.update([1]);
            hash.update(value.to_bits().to_le_bytes());
        }
        NumericScalar::F32(value) => {
            hash.update([2]);
            hash.update(value.to_bits().to_le_bytes());
        }
        NumericScalar::I8(value) => {
            hash.update([3]);
            hash.update(value.to_le_bytes());
        }
        NumericScalar::I16(value) => {
            hash.update([4]);
            hash.update(value.to_le_bytes());
        }
        NumericScalar::I32(value) => {
            hash.update([5]);
            hash.update(value.to_le_bytes());
        }
        NumericScalar::I64(value) => {
            hash.update([6]);
            hash.update(value.to_le_bytes());
        }
        NumericScalar::U8(value) => {
            hash.update([7]);
            hash.update(value.to_le_bytes());
        }
        NumericScalar::U16(value) => {
            hash.update([8]);
            hash.update(value.to_le_bytes());
        }
        NumericScalar::U32(value) => {
            hash.update([9]);
            hash.update(value.to_le_bytes());
        }
        NumericScalar::U64(value) => {
            hash.update([10]);
            hash.update(value.to_le_bytes());
        }
    }
    Some(())
}
