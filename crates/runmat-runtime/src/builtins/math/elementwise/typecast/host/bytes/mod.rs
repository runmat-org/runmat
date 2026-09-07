mod complex;
mod decode;
mod encode;

pub(super) fn encode(source: runmat_value::Value) -> crate::BuiltinResult<Vec<u8>> {
    encode::value(source)
}

pub(super) fn decode(
    bytes: &[u8],
    target: super::super::target::Representation,
) -> runmat_value::NumericStorage {
    decode::storage(bytes, target)
}

pub(super) fn pair_complex(storage: runmat_value::NumericStorage) -> runmat_value::ComplexStorage {
    complex::pair(storage)
}
