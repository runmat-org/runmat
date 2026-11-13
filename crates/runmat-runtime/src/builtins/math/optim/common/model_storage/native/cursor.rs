//! The same quote/promotion loops serve the owned publication cursor and
//! small existing-owner cursors. These are borrowed views, not another ledger.
use super::*;
pub(super) struct Quote<'a> {
    pub storage: &'a Collections,
    pub seen: &'a mut Map<SharedStorageIdentity, traversal::Seen>,
    pub policy: Policy,
    pub context: &'a RuntimeContext,
    pub bytes: usize,
    pub claims: usize,
}
impl Quote<'_> {
    pub async fn cursor<S: OptimizationNativeSource>(
        &mut self,
        cursor: &mut OptimizationNativeAdmission<S>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        while let Some(quote) = cursor
            .quote_next()
            .map_err(|source| metadata(source, self.policy, Some(self.context)))?
        {
            if let Some(count) = quote.bytes {
                let id = quote
                    .owner
                    .ok_or_else(|| allocation(Some(self.context), HostAllocationError::Overflow))?;
                if !self.seen.get(&id).is_some_and(|value| value.quoted) {
                    let additional = usize::from(!self.seen.contains_key(&id));
                    self.storage.reserve_map(self.seen, additional)?;
                    self.seen.entry(id).or_default().quoted = true;
                    self.bytes = self.bytes.checked_add(count).ok_or_else(|| {
                        allocation(Some(self.context), HostAllocationError::Overflow)
                    })?;
                    self.claims = self
                        .claims
                        .checked_add(usize::from(count != 0))
                        .ok_or_else(|| {
                            allocation(Some(self.context), HostAllocationError::Overflow)
                        })?;
                }
            }
            cooperation.completed(1).await?;
            self.storage.check()?;
        }
        Ok(())
    }
    pub fn totals(self) -> (usize, usize) {
        (self.bytes, self.claims)
    }
}
pub(super) struct Promotion<'a> {
    pub storage: &'a Collections,
    pub policy: Policy,
    pub context: &'a RuntimeContext,
    pub receipt: &'a HostAllocationReceipt,
}
impl Promotion<'_> {
    pub async fn cursor<S: OptimizationNativeSource>(
        &self,
        cursor: &mut OptimizationNativeAdmission<S>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        cursor
            .begin_admission()
            .map_err(|source| metadata(source, self.policy, Some(self.context)))?;
        loop {
            cooperation.check_allocation_epoch()?;
            let advanced = self
                .receipt
                .with_output_construction(self.context, |permit| cursor.advance_admission(permit))
                .map_err(|source| allocation(Some(self.context), source))?
                .map_err(|source| metadata(source, self.policy, Some(self.context)))?;
            if !advanced {
                break;
            }
            cooperation.completed(1).await?;
            self.storage.check()?;
        }
        Ok(())
    }
}
