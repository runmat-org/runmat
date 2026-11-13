//! CBOR header widths shared with the encoder, without allocating a wire buffer.
use std::convert::Infallible;

#[derive(Default)]
struct Count(u64);
impl minicbor::encode::Write for Count {
    type Error = Infallible;
    fn write_all(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.0 += bytes.len() as u64;
        Ok(())
    }
}

pub(super) fn header(value: u64) -> u64 {
    let mut encoder = minicbor::Encoder::new(Count::default());
    encoder.u64(value).expect("infallible scalar counter");
    encoder.into_writer().0
}

pub(super) fn integer(value: i128) -> u64 {
    if value >= 0 {
        header(value as u64)
    } else {
        let mut encoder = minicbor::Encoder::new(Count::default());
        encoder
            .i64(value as i64)
            .expect("infallible scalar counter");
        encoder.into_writer().0
    }
}
