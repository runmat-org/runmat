use runmat_runtime::object::protocol::PreparedSubscriptReceiver;

use super::{super::HostState, EmbeddedOperationIdentity};
use crate::NativeExecutorResult;

impl HostState {
    pub fn cached_prepared_subscript_receiver(
        &self,
        owner: Option<&EmbeddedOperationIdentity>,
        step: usize,
    ) -> Option<PreparedSubscriptReceiver> {
        self.contextual_regions
            .metadata
            .as_ref()?
            .prepared_subscript_receivers
            .get(&(owner.cloned(), step))
            .map(|(prepared, _)| prepared.clone())
    }

    pub fn cache_prepared_subscript_receiver(
        &mut self,
        owner: Option<EmbeddedOperationIdentity>,
        step: usize,
        prepared: PreparedSubscriptReceiver,
    ) -> NativeExecutorResult<()> {
        let metadata = self.contextual_regions.metadata_mut()?;
        if metadata
            .prepared_subscript_receivers
            .get(&(owner.clone(), step))
            .is_none()
        {
            metadata
                .prepared_subscript_receivers
                .prepare_one_more(&self.runtime, &metadata.gate)?;
        }
        let receiver = self
            .arena
            .insert(prepared.receiver().clone(), Some(&self.runtime))?;
        let previous = self
            .contextual_regions
            .metadata_mut()?
            .prepared_subscript_receivers
            .insert_ready((owner, step), (prepared, receiver));
        if let Some((_, old_receiver)) = previous {
            self.arena.release(old_receiver)?;
        }
        Ok(())
    }
}
