//! Allocation-free canonical grammar admission. All headers use their shortest
//! form; finite floats use the established f64 projection, including signed zero.
use super::*;
use minicbor::{data::Type, Decoder};

struct Scan<'a, 'c> {
    decoder: Decoder<'a>,
    limits: CanonicalLimits,
    control: &'c dyn CanonicalCodecControl,
}
pub(super) fn validate(
    domain: &str,
    payload: &[u8],
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<usize, CanonicalCodecError> {
    let mut scan = Scan {
        decoder: Decoder::new(payload),
        limits,
        control,
    };
    let start = scan.decoder.position();
    if scan.decoder.array().map_err(invalid)? != Some(2) {
        return Err(invalid(
            "canonical envelope must be a definite two-element array",
        ));
    }
    scan.header(start, 2)?;
    let start = scan.decoder.position();
    let actual = scan.decoder.str().map_err(invalid)?;
    scan.header(start, actual.len() as u64)?;
    if actual != domain {
        return Err(CanonicalCodecError::invalid(
            "canonical decoding domain",
            format!("expected {domain}, received {actual}"),
        ));
    }
    let value_start = scan.decoder.position();
    scan.value(0)?;
    if scan.decoder.position() != payload.len() {
        return Err(invalid("trailing data is forbidden"));
    }
    Ok(value_start)
}
impl<'a> Scan<'a, '_> {
    fn header(&self, start: usize, argument: u64) -> Result<(), CanonicalCodecError> {
        let expected = match argument {
            0..=23 => argument as u8,
            24..=255 => 24,
            256..=65535 => 25,
            65536..=0xffff_ffff => 26,
            _ => 27,
        };
        if self.decoder.input()[start] & 31 != expected {
            return Err(invalid("input is not in canonical form"));
        }
        Ok(())
    }
    fn string(&mut self) -> Result<&'a str, CanonicalCodecError> {
        let start = self.decoder.position();
        let value = self.decoder.str().map_err(invalid)?;
        self.header(start, value.len() as u64)?;
        if value.len() > self.limits.maximum_string_bytes {
            return Err(invalid("string exceeds the contract limit"));
        }
        Ok(value)
    }
    fn collection(&self, start: usize, count: Option<u64>) -> Result<usize, CanonicalCodecError> {
        let count = count.ok_or_else(|| invalid("indefinite collections are forbidden"))?;
        self.header(start, count)?;
        let count = usize::try_from(count).map_err(invalid)?;
        if count > self.limits.maximum_collection_items {
            return Err(invalid("collection exceeds the contract limit"));
        }
        Ok(count)
    }
    fn value(&mut self, depth: usize) -> Result<(), CanonicalCodecError> {
        checkpoint(self.control)?;
        if depth > self.limits.maximum_nesting_depth {
            return Err(invalid("nesting depth exceeds the contract limit"));
        }
        let start = self.decoder.position();
        match self.decoder.datatype().map_err(invalid)? {
            Type::Null => {
                self.decoder.null().map_err(invalid)?;
            }
            Type::Bool => {
                self.decoder.bool().map_err(invalid)?;
            }
            Type::U8 | Type::U16 | Type::U32 | Type::U64 => {
                let value = self.decoder.u64().map_err(invalid)?;
                self.header(start, value)?;
            }
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Int => {
                let value = self.decoder.i64().map_err(invalid)?;
                self.header(start, (-1_i128 - i128::from(value)) as u64)?;
            }
            Type::F64 => {
                if !self.decoder.f64().map_err(invalid)?.is_finite() {
                    return Err(invalid("non-finite floating point values are forbidden"));
                }
            }
            Type::String => {
                self.string()?;
            }
            Type::Array => {
                let length = self.decoder.array().map_err(invalid)?;
                let count = self.collection(start, length)?;
                for _ in 0..count {
                    self.value(depth + 1)?;
                }
            }
            Type::Map => {
                let length = self.decoder.map().map_err(invalid)?;
                let count = self.collection(start, length)?;
                let mut previous: Option<&str> = None;
                for _ in 0..count {
                    checkpoint(self.control)?;
                    let key = self.string()?;
                    if previous.is_some_and(|previous| previous.as_bytes() >= key.as_bytes()) {
                        return Err(invalid(
                            "map keys must be unique and ordered by UTF-8 bytes",
                        ));
                    }
                    previous = Some(key);
                    self.value(depth + 1)?;
                }
            }
            _ => return Err(invalid("unsupported or noncanonical CBOR representation")),
        }
        Ok(())
    }
}
