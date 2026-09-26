use super::{EmbeddedOperationIdentity, EmbeddedOperationSegment, OperationScope};
use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::HostState;

impl HostState {
    pub fn enter_embedded_call(&mut self) -> Option<EmbeddedOperationIdentity> {
        let scope = self.contextual_regions.operation_path.last_mut()?;
        let ordinal = scope.next_call;
        scope.next_call += 1;
        self.contextual_regions.operation_path.push(OperationScope {
            segment: EmbeddedOperationSegment::Call { ordinal },
            next_call: 0,
        });
        self.current_embedded_operation()
    }

    pub fn finish_embedded_call(
        &mut self,
        identity: Option<&EmbeddedOperationIdentity>,
    ) -> NativeExecutorResult<()> {
        let Some(identity) = identity else {
            return Ok(());
        };
        let Some(EmbeddedOperationSegment::Call { ordinal }) = identity.path.last() else {
            return Err(NativeExecutorError::Host(
                "embedded call identity has no call segment".into(),
            ));
        };
        self.contextual_regions
            .operation_region_ends
            .insert(identity.clone(), self.contextual_regions.next_region);
        self.pop_operation(EmbeddedOperationSegment::Call { ordinal: *ordinal })
    }

    pub fn restore_embedded_call_regions(&mut self, identity: &EmbeddedOperationIdentity) {
        if let Some(next_region) = self.contextual_regions.operation_region_ends.get(identity) {
            self.contextual_regions.next_region = *next_region;
        }
    }

    pub fn current_embedded_operation(&self) -> Option<EmbeddedOperationIdentity> {
        (!self.contextual_regions.operation_path.is_empty()).then(|| EmbeddedOperationIdentity {
            path: self
                .contextual_regions
                .operation_path
                .iter()
                .map(|scope| scope.segment.clone())
                .collect(),
        })
    }

    pub fn cached_embedded_call(
        &mut self,
        identity: &EmbeddedOperationIdentity,
    ) -> NativeExecutorResult<Option<Vec<runmat_value::Value>>> {
        let cached = self
            .contextual_regions
            .completed_operations
            .get(identity)
            .map(|references| {
                references
                    .iter()
                    .map(|reference| self.arena.get(*reference).cloned())
                    .collect::<NativeExecutorResult<Vec<_>>>()
            })
            .transpose()?;
        if cached.is_some() {
            self.restore_embedded_call_regions(identity);
        }
        Ok(cached)
    }

    pub fn cache_embedded_call(
        &mut self,
        identity: Option<&EmbeddedOperationIdentity>,
        values: &[runmat_value::Value],
    ) {
        let Some(identity) = identity else {
            return;
        };
        let references = values
            .iter()
            .cloned()
            .map(|value| self.arena.insert(value))
            .collect();
        self.contextual_regions
            .completed_operations
            .insert(identity.clone(), references);
    }
}

impl EmbeddedOperationIdentity {
    pub fn is_descendant_of(&self, ancestor: Option<&Self>) -> bool {
        match ancestor {
            None => !self.path.is_empty(),
            Some(ancestor) => {
                self.path.len() > ancestor.path.len()
                    && self.path.starts_with(ancestor.path.as_slice())
            }
        }
    }
}
