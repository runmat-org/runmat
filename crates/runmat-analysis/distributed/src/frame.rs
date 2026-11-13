//! Compact little-endian binary frames for rank inputs and outputs.

use crate::error::{DistributedError, DistributedResult};

#[derive(Default)]
pub(crate) struct Encoder {
    pub bytes: Vec<u8>,
}

impl Encoder {
    pub fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn len(&mut self, length: usize) {
        self.u64(length as u64);
    }

    pub fn u32s(&mut self, values: &[u32]) {
        self.len(values.len());
        self.bytes.reserve(values.len() * 4);
        for value in values {
            self.u32(*value);
        }
    }

    pub fn u64s(&mut self, values: &[u64]) {
        self.len(values.len());
        self.bytes.reserve(values.len() * 8);
        for value in values {
            self.u64(*value);
        }
    }

    pub fn f64s(&mut self, values: &[f64]) {
        self.len(values.len());
        self.bytes.reserve(values.len() * 8);
        for value in values {
            self.f64(*value);
        }
    }

    pub fn bools(&mut self, values: &[bool]) {
        self.len(values.len());
        self.bytes
            .extend(values.iter().map(|&value| u8::from(value)));
    }

    pub fn points(&mut self, values: &[[f64; 3]]) {
        self.len(values.len());
        self.bytes.reserve(values.len() * 24);
        for point in values {
            for value in point {
                self.f64(*value);
            }
        }
    }

    pub fn pairs(&mut self, values: &[(u32, f64)]) {
        self.len(values.len());
        for (index, value) in values {
            self.u32(*index);
            self.f64(*value);
        }
    }
}

pub(crate) struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> DistributedResult<&'a [u8]> {
        let end = self
            .position
            .checked_add(count)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| DistributedError::Communication("truncated rank frame".into()))?;
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    pub fn finish(&self) -> DistributedResult<()> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(DistributedError::Communication(
                "rank frame has trailing bytes".into(),
            ))
        }
    }

    pub fn u32(&mut self) -> DistributedResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    pub fn u64(&mut self) -> DistributedResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    pub fn f64(&mut self) -> DistributedResult<f64> {
        Ok(f64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    pub fn len(&mut self, element_bytes: usize) -> DistributedResult<usize> {
        let length = self.u64()? as usize;
        if length
            .checked_mul(element_bytes)
            .is_none_or(|bytes| bytes > self.bytes.len() - self.position)
        {
            return Err(DistributedError::Communication(
                "rank frame length exceeds its payload".into(),
            ));
        }
        Ok(length)
    }

    pub fn u32s(&mut self) -> DistributedResult<Vec<u32>> {
        let length = self.len(4)?;
        (0..length).map(|_| self.u32()).collect()
    }

    pub fn u64s(&mut self) -> DistributedResult<Vec<u64>> {
        let length = self.len(8)?;
        (0..length).map(|_| self.u64()).collect()
    }

    pub fn f64s(&mut self) -> DistributedResult<Vec<f64>> {
        let length = self.len(8)?;
        (0..length).map(|_| self.f64()).collect()
    }

    pub fn bools(&mut self) -> DistributedResult<Vec<bool>> {
        let length = self.len(1)?;
        Ok(self.take(length)?.iter().map(|&byte| byte != 0).collect())
    }

    pub fn points(&mut self) -> DistributedResult<Vec<[f64; 3]>> {
        let length = self.len(24)?;
        (0..length)
            .map(|_| Ok([self.f64()?, self.f64()?, self.f64()?]))
            .collect()
    }

    pub fn pairs(&mut self) -> DistributedResult<Vec<(u32, f64)>> {
        let length = self.len(12)?;
        (0..length)
            .map(|_| Ok((self.u32()?, self.f64()?)))
            .collect()
    }
}
