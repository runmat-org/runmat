//! Pre-allocation charges use the same aggregate counters as final validation.
use super::*;
use serde::de::DeserializeOwned;

impl Budget {
    pub(super) fn nodes(&mut self, count: u64) -> BuiltinResult<()> {
        self.usage.nodes(count).map_err(limit)
    }
    pub(super) fn elements(&mut self, count: u64) -> BuiltinResult<()> {
        self.usage.elements(count).map_err(limit)
    }
    pub(super) fn backing(&mut self, bytes: u64) -> BuiltinResult<()> {
        self.usage.bytes(bytes).map_err(limit)
    }
    pub(super) async fn text(
        &mut self,
        reader: &mut Reader,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<String> {
        c.completed(1).await?;
        let remaining = self.usage.remaining().max_text_bytes;
        let value = reader
            .string_limited(usize::try_from(remaining).unwrap_or(usize::MAX), c)
            .await?;
        self.usage.text(value.len() as u64).map_err(limit)?;
        Ok(value)
    }
    pub(super) async fn optional_text(
        &mut self,
        reader: &mut Reader,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Option<String>> {
        if reader.null()? {
            Ok(None)
        } else {
            self.text(reader, c).await.map(Some)
        }
    }
    pub(super) async fn array<T: DeserializeOwned>(
        &mut self,
        reader: &mut Reader,
        maximum_item: usize,
        elements: u64,
        bytes: u64,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Vec<T>> {
        reader.token(b'[')?;
        let mut values = Vec::new();
        while !reader.is(b']') {
            self.elements(elements)?;
            self.backing(bytes)?;
            c.completed(1).await?;
            if !values.is_empty() {
                reader.token(b',')?;
            }
            values
                .try_reserve(1)
                .map_err(|source| with_source("selected graph array allocation failed", source))?;
            values.push(reader.metadata(maximum_item, c).await?);
        }
        reader.token(b']')?;
        Ok(values)
    }
    pub(super) async fn shape(
        &mut self,
        reader: &mut Reader,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Vec<u64>> {
        self.array(reader, 20, 0, 8, c).await
    }
    pub(super) async fn references(
        &mut self,
        reader: &mut Reader,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Vec<GraphNodeRef>> {
        self.array(reader, 20, 1, 4, c).await
    }
    pub(super) async fn bytes(
        &mut self,
        reader: &mut Reader,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Vec<u8>> {
        self.array(reader, 3, 0, 1, c).await
    }
}
fn limit(source: runmat_execution::ContractError) -> crate::RuntimeError {
    with_source("selected graph exceeds its shared value limits", source)
}
