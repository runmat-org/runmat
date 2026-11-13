use super::*;

impl ValuePayloadValidationCursor<'_> {
    pub(super) fn reference(&mut self, value: &ValueRef) -> Result<(), ContractError> {
        if value.schema_version != VALUE_PAYLOAD_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: value.schema_version,
                supported: VALUE_PAYLOAD_SCHEMA_VERSION,
            });
        }
        self.budget.add_text(&value.media_type)?;
        self.budget.add_text(&value.value_schema)?;
        self.budget.add_text(&value.authorization_scope)?;
        match (value.kind, &value.resident_fence) {
            (ValueRefKind::ResidentObject, None) => Err(invalid(
                "resident value",
                "resident references require a worker/node/process fence",
            )),
            (ValueRefKind::ResidentObject, Some(fence)) => {
                fence.validate()?;
                match self.budget.transport {
                    None => Ok(()),
                    Some(ValueTransportContext::Portable) => Err(invalid("resident value", "process- or device-resident references cannot cross a portable transport boundary")),
                    Some(ValueTransportContext::Resident(authority)) if authority.authorizes(fence) => Ok(()),
                    Some(ValueTransportContext::Resident(_)) => Err(invalid("resident value", "worker, node lease, process generation, attempt, or accelerator lease is stale or different")),
                }
            }
            (_, Some(_)) => Err(invalid(
                "value reference",
                "only resident references may carry a resident fence",
            )),
            (_, None) => Ok(()),
        }
    }

    pub(super) fn class(
        &mut self,
        value: &runmat_types::ClassIdentity,
    ) -> Result<(), ContractError> {
        runmat_types::ClassIdentity::new(value.display_name())
            .map_err(|_| invalid("class identity", "invalid class name"))?;
        self.budget.add_text(value.display_name())
    }

    pub(super) fn field_limit(
        &self,
        count: usize,
        label: &'static str,
    ) -> Result<(), ContractError> {
        if count > self.budget.limits.max_fields as usize {
            Err(ContractError::Limit {
                field: label,
                limit: self.budget.limits.max_fields.into(),
            })
        } else {
            Ok(())
        }
    }
}
