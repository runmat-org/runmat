use runmat_value::{ComplexStorage, IntegerComplexStorage, IntegerStorage, NumericStorage};

pub(super) fn pair(storage: NumericStorage) -> ComplexStorage {
    macro_rules! integer {
        ($values:expr, $variant:ident) => {{
            let (real, imaginary) = lanes($values);
            ComplexStorage::Integer(
                IntegerComplexStorage::new(
                    IntegerStorage::$variant(real),
                    IntegerStorage::$variant(imaginary),
                )
                .expect("paired typecast storage has matching class and length"),
            )
        }};
    }
    match storage {
        NumericStorage::F64(values) => {
            let (real, imaginary) = lanes(values);
            ComplexStorage::F64(real.into_iter().zip(imaginary).collect())
        }
        NumericStorage::F32(values) => {
            let (real, imaginary) = lanes(values);
            ComplexStorage::F32(real.into_iter().zip(imaginary).collect())
        }
        NumericStorage::I8(values) => integer!(values, I8),
        NumericStorage::I16(values) => integer!(values, I16),
        NumericStorage::I32(values) => integer!(values, I32),
        NumericStorage::I64(values) => integer!(values, I64),
        NumericStorage::U8(values) => integer!(values, U8),
        NumericStorage::U16(values) => integer!(values, U16),
        NumericStorage::U32(values) => integer!(values, U32),
        NumericStorage::U64(values) => integer!(values, U64),
    }
}

fn lanes<T>(values: Vec<T>) -> (Vec<T>, Vec<T>) {
    let mut real = Vec::with_capacity(values.len() / 2);
    let mut imaginary = Vec::with_capacity(values.len() / 2);
    let mut values = values.into_iter();
    while let Some(value) = values.next() {
        real.push(value);
        imaginary.push(values.next().expect("complex byte count was validated"));
    }
    (real, imaginary)
}
