//! Admit a class-only dependency into one exact selected graph unit bundle.

use std::collections::HashMap;
use std::sync::Arc;

use runmat_execution::value::{GraphProgramUnitId, GRAPH_MAX_SELECTED_UNITS};
use runmat_runtime::execution::value_codec::{GraphSourceUnitFacts, ValueCodecError};

use super::{checked_interop_receipt, same_receipt, SelectedGraphUnit, SelectedGraphUnits};
use crate::session::source_evaluation::selected_graph::selection::{
    invalid, SelectedDeclarationBody,
};

impl SelectedGraphUnits {
    pub(in crate::session::source_evaluation::selected_graph) fn add_declaration(
        &mut self,
        declaration: &SelectedDeclarationBody,
    ) -> Result<GraphProgramUnitId, ValueCodecError> {
        if !declaration.source_owner.is_live() {
            return Err(invalid(
                "selected class declaration source owner has retired",
            ));
        }
        if let Some((position, unit)) = self
            .units
            .iter()
            .enumerate()
            .find(|(_, unit)| unit.source_owner.same_issuance(&declaration.source_owner))
        {
            if unit.bytes.as_ref() != declaration.bytes.as_ref() {
                return Err(invalid(
                    "one selected source issuance has conflicting class bytes",
                ));
            }
            if !same_receipt(
                unit.interop_receipt.as_ref(),
                declaration.interop_receipt.as_ref(),
            ) {
                return Err(invalid(
                    "one selected source issuance has conflicting class foreign receipts",
                ));
            }
            return u16::try_from(position)
                .map(GraphProgramUnitId)
                .map_err(|_| invalid("selected graph unit ordinal exceeds u16"));
        }
        if self.units.len() >= GRAPH_MAX_SELECTED_UNITS {
            return Err(invalid("selected graph exceeds its unit bound"));
        }
        let admitted = runmat_vm::admit_executable_unit(&declaration.bytes)
            .map_err(|_| invalid("selected class declaration unit is invalid"))?;
        if admitted.declaration_class() != Some(&declaration.class)
            || declaration.source_owner.portable_map() != admitted.admitted_source_map()
        {
            return Err(invalid(
                "selected class declaration differs from its source owner",
            ));
        }
        checked_interop_receipt(
            &admitted,
            &declaration.bytes,
            &declaration.source_owner,
            declaration.interop_receipt.as_ref(),
        )?;
        let revision = &admitted.envelope().manifest.identity.program;
        let facts = GraphSourceUnitFacts::capture_owner(
            declaration.source_owner.clone(),
            revision,
            Arc::new(admitted.admitted_source_map().clone()),
        )?;
        let total_bytes = self
            .total_bytes
            .checked_add(declaration.bytes.len())
            .ok_or_else(|| invalid("selected graph unit bytes overflow"))?;
        if total_bytes > runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES {
            return Err(invalid(
                "selected graph unit bytes exceed their shared bound",
            ));
        }
        let position = self.units.len();
        self.units.push(SelectedGraphUnit {
            bytes: Arc::clone(&declaration.bytes),
            facts,
            source_owner: declaration.source_owner.clone(),
            fingerprints: HashMap::new(),
            interop_receipt: declaration.interop_receipt.clone(),
        });
        self.total_bytes = total_bytes;
        u16::try_from(position)
            .map(GraphProgramUnitId)
            .map_err(|_| invalid("selected graph unit ordinal exceeds u16"))
    }
}
