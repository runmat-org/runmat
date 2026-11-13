//! Allocation-free serialization of canonical live value facts. This is a
//! borrowed view, not a second fact schema or an execution memory certificate.
use super::*;
use crate::call::FunctionArgumentView;
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Serialize, Serializer};

/// A supported argument window whose serialized representation is exactly
/// `arguments.iter().map(value_fact).collect::<Vec<_>>()`. Shape extents and
/// Values remain borrowed from their original owners. Recursive fact joins,
/// callable identities and foreign metadata require a separate funded owner;
/// they decline here before serializing or allocating anything.
#[derive(Clone, Copy)]
pub struct BorrowedValueFacts<'a>(Inputs<'a>);
#[derive(Clone, Copy)]
enum Inputs<'a> {
    Arguments(FunctionArgumentView<'a>),
    References(&'a [&'a Value]),
}

impl<'a> BorrowedValueFacts<'a> {
    pub fn new(arguments: FunctionArgumentView<'a>) -> Option<Self> {
        arguments
            .iter()
            .all(|value| BorrowedValueFact::new(value).is_some())
            .then_some(Self(Inputs::Arguments(arguments)))
    }
    /// Native numeric frames already own bounded borrowed slots; do not copy
    /// Values or invent a second fact serialization to hash those slots.
    pub fn from_references(values: &'a [&'a Value]) -> Option<Self> {
        values
            .iter()
            .all(|value| BorrowedValueFact::new(value).is_some())
            .then_some(Self(Inputs::References(values)))
    }
}

impl Serialize for BorrowedValueFacts<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let len = match self.0 {
            Inputs::Arguments(arguments) => arguments.len(),
            Inputs::References(values) => values.len(),
        };
        let mut sequence = serializer.serialize_seq(Some(len))?;
        for index in 0..len {
            let value = match self.0 {
                Inputs::Arguments(arguments) => {
                    arguments.get(index).expect("checked argument index")
                }
                Inputs::References(values) => values[index],
            };
            let fact = BorrowedValueFact::new(value)
                .ok_or_else(|| serde::ser::Error::custom("unsupported borrowed value fact"))?;
            sequence.serialize_element(&fact)?;
        }
        sequence.end()
    }
}

enum ShapeDimensions<'a> {
    Scalar,
    Borrowed(&'a [usize]),
    Sparse([usize; 2]),
}

struct BorrowedValueFact<'a> {
    // Scalar construction allocates neither a shape vector nor a fact tree.
    fact: ValueFact,
    dimensions: ShapeDimensions<'a>,
}

impl<'a> BorrowedValueFact<'a> {
    fn new(value: &'a Value) -> Option<Self> {
        let (kind, dimensions, storage) = match value {
            Value::Int(value) => (
                numeric_kind(int_class(value), NumericDomain::Real),
                ShapeDimensions::Scalar,
                StorageFact::Scalar,
            ),
            Value::Num(_) => (
                numeric_kind(NumericClass::Double, NumericDomain::Real),
                ShapeDimensions::Scalar,
                StorageFact::Scalar,
            ),
            Value::Complex(_, _) => (
                numeric_kind(NumericClass::Double, NumericDomain::Complex),
                ShapeDimensions::Scalar,
                StorageFact::Scalar,
            ),
            Value::Bool(_) => (
                ValueKindFact::Logical,
                ShapeDimensions::Scalar,
                StorageFact::Scalar,
            ),
            Value::String(_) => (
                ValueKindFact::String,
                ShapeDimensions::Scalar,
                StorageFact::Scalar,
            ),
            Value::Symbolic(_) => (
                ValueKindFact::Symbolic,
                ShapeDimensions::Scalar,
                StorageFact::Scalar,
            ),
            Value::Tensor(value) => (
                numeric_kind(dtype_class(value.numeric_dtype()), NumericDomain::Real),
                ShapeDimensions::Borrowed(value.shape()),
                StorageFact::Dense,
            ),
            Value::ComplexTensor(value) => (
                numeric_kind(dtype_class(value.numeric_dtype()), NumericDomain::Complex),
                ShapeDimensions::Borrowed(value.shape()),
                StorageFact::Dense,
            ),
            Value::SparseTensor(value) => (
                value
                    .numeric_dtype()
                    .map_or(ValueKindFact::Logical, |dtype| {
                        numeric_kind(
                            dtype_class(dtype),
                            if value.is_complex() {
                                NumericDomain::Complex
                            } else {
                                NumericDomain::Real
                            },
                        )
                    }),
                ShapeDimensions::Sparse([value.rows, value.cols]),
                StorageFact::Sparse,
            ),
            Value::LogicalArray(value) => (
                ValueKindFact::Logical,
                ShapeDimensions::Borrowed(&value.shape),
                StorageFact::Dense,
            ),
            Value::StringArray(value) => (
                ValueKindFact::String,
                ShapeDimensions::Borrowed(&value.shape),
                StorageFact::Dense,
            ),
            Value::CharArray(value) => (
                ValueKindFact::Character,
                ShapeDimensions::Borrowed(&value.shape),
                StorageFact::Dense,
            ),
            Value::SymbolicArray(value) => (
                ValueKindFact::Symbolic,
                ShapeDimensions::Borrowed(&value.shape),
                StorageFact::Dense,
            ),
            Value::Cell(_)
            | Value::Struct(_)
            | Value::StructArray(_)
            | Value::GpuTensor(_)
            | Value::Object(_)
            | Value::ObjectArray(_)
            | Value::HandleObject(_)
            | Value::Listener(_)
            | Value::OutputList(_)
            | Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
            | Value::ClassRef(_)
            | Value::MException(_)
            | Value::Future(_)
            | Value::Task(_)
            | Value::Pool(_)
            | Value::Job(_)
            | Value::Distributed(_)
            | Value::Composite(_)
            | Value::Foreign(_) => return None,
        };
        Some(Self {
            fact: fact(kind, ShapeFact::Scalar, storage),
            dimensions,
        })
    }
}

impl Serialize for BorrowedValueFact<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Field order and enum representation intentionally match ValueFact's
        // canonical derived Serialize; parity tests protect the byte identity.
        let mut fields = serializer.serialize_struct("ValueFact", 11)?;
        fields.serialize_field("kind", &self.fact.kind)?;
        match &self.dimensions {
            ShapeDimensions::Scalar => fields.serialize_field("shape", &ShapeFact::Scalar)?,
            ShapeDimensions::Borrowed(dims) => {
                fields.serialize_field("shape", &BorrowedShape { dims })?
            }
            ShapeDimensions::Sparse(dims) => {
                fields.serialize_field("shape", &BorrowedShape { dims })?
            }
        }
        fields.serialize_field("storage", &self.fact.storage)?;
        fields.serialize_field("layout", &self.fact.layout)?;
        fields.serialize_field("contiguity", &self.fact.contiguity)?;
        fields.serialize_field("view", &self.fact.view)?;
        fields.serialize_field("residency", &self.fact.residency)?;
        fields.serialize_field("alias", &self.fact.alias)?;
        fields.serialize_field("mutation", &self.fact.mutation)?;
        fields.serialize_field("certainty", &self.fact.certainty)?;
        fields.serialize_field("invalidation", &self.fact.invalidation)?;
        fields.end()
    }
}

struct BorrowedShape<'a> {
    dims: &'a [usize],
}
impl Serialize for BorrowedShape<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStructVariant;
        let mut shape = serializer.serialize_struct_variant("ShapeFact", 3, "Shaped", 1)?;
        shape.serialize_field("dims", &KnownDimensions(self.dims))?;
        shape.end()
    }
}
struct KnownDimensions<'a>(&'a [usize]);
impl Serialize for KnownDimensions<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut dims = serializer.serialize_seq(Some(self.0.len()))?;
        for dim in self.0 {
            dims.serialize_element(&DimensionFact::Known(*dim))?;
        }
        dims.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::Digest;
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, LogicalArray, SparseTensor, StringArray, Tensor,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn borrowed_facts_preserve_canonical_bytes_and_decline_unfunded_trees() {
        let mut values = vec![
            Value::Num(1.0),
            Value::Complex(2.0, 3.0),
            Value::Bool(true),
            Value::String("not fact payload".into()),
        ];
        for dims in [vec![], vec![1, 1], vec![0, 2], vec![2, 3]] {
            let len = dims.iter().product();
            values.push(Value::Tensor(
                Tensor::from_f32(vec![0.0; len], dims.clone()).unwrap(),
            ));
            values.push(Value::ComplexTensor(
                ComplexTensor::new(vec![(0.0, 1.0); len], dims).unwrap(),
            ));
        }
        values.extend([
            Value::SparseTensor(SparseTensor::zeros(0, 3)),
            Value::SparseTensor(SparseTensor::zeros_complex_f32(2, 3)),
            Value::Int(IntValue::I8(-1)),
            Value::Int(IntValue::I16(-2)),
            Value::Int(IntValue::I32(-3)),
            Value::Int(IntValue::I64(-4)),
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(2)),
            Value::Int(IntValue::U32(3)),
            Value::Int(IntValue::U64(4)),
            Value::CharArray(CharArray::new_row("x💡")),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
            Value::StringArray(StringArray::new(vec!["x".into()], vec![1, 1]).unwrap()),
        ]);
        let borrowed = BorrowedValueFacts::new(FunctionArgumentView::from_slice(&values)).unwrap();
        let owned: Vec<_> = values.iter().map(value_fact).collect();
        assert_eq!(
            serde_json::to_vec(&borrowed).unwrap(),
            serde_json::to_vec(&owned).unwrap()
        );
        assert_eq!(
            Digest::sha256_serialized(&borrowed).unwrap(),
            Digest::sha256_serialized(&owned).unwrap()
        );
        let references: Vec<_> = values.iter().collect();
        let native = BorrowedValueFacts::from_references(&references).unwrap();
        assert_eq!(
            serde_json::to_vec(&native).unwrap(),
            serde_json::to_vec(&owned).unwrap()
        );
        assert_eq!(
            Digest::sha256_serialized(&native).unwrap(),
            Digest::sha256_serialized(&owned).unwrap()
        );
        let nested = [Value::Cell(
            CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap(),
        )];
        assert!(BorrowedValueFacts::new(FunctionArgumentView::from_slice(&nested)).is_none());
        assert!(BorrowedValueFacts::from_references(&[&nested[0]]).is_none());
        assert!(BorrowedValueFacts::new(FunctionArgumentView::from_slice(&[])).is_some());
    }
}
