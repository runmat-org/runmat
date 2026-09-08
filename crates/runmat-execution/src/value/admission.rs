use runmat_types::{
    AliasFact, CertaintyFact, ContiguityFact, LayoutFact, MutationFact, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};
use serde::{Deserialize, Serialize};

use crate::ContractError;

pub const VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION: u16 = 1;

/// How a live value's storage became owned by a portable execution input.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueAdmissionOwnership {
    Borrowed,
    Moved,
    Materialized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueConversionFidelity {
    Exact,
    PotentiallyRounded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueNumericConversion {
    pub source: NumericClass,
    pub target: NumericClass,
    pub fidelity: ValueConversionFidelity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueAdmissionTransfer {
    None,
    DeviceToHost { device_id: u32 },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueCopyAccounting {
    /// Bytes allocated and written to change the host representation or ownership.
    pub host_materialization_bytes: u64,
    /// Physical bytes copied from provider-owned device storage to host storage.
    pub device_to_host_bytes: u64,
}

/// Portable evidence emitted when runtime-owned storage enters an execution domain.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueAdmissionReceipt {
    pub schema_version: u16,
    pub source: ValueFact,
    pub target: ValueFact,
    pub ownership: ValueAdmissionOwnership,
    pub conversion: Option<ValueNumericConversion>,
    pub transfer: ValueAdmissionTransfer,
    pub copies: ValueCopyAccounting,
}

impl ValueAdmissionReceipt {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
            });
        }
        if self.target.certainty != CertaintyFact::Proven {
            return Err(invalid("admitted value target is not proven"));
        }
        match self.transfer {
            ValueAdmissionTransfer::None if self.copies.device_to_host_bytes != 0 => {
                return Err(invalid("device copy bytes exist without a transfer"));
            }
            ValueAdmissionTransfer::DeviceToHost { .. }
                if !matches!(self.source.residency, ResidencyFact::Device { .. }) =>
            {
                return Err(invalid("device transfer has a non-device source"));
            }
            ValueAdmissionTransfer::DeviceToHost { .. }
                if self.target.residency != ResidencyFact::Host
                    || self.copies.device_to_host_bytes == 0 =>
            {
                return Err(invalid("device transfer has no exact host copy evidence"));
            }
            _ => {}
        }
        if self.ownership == ValueAdmissionOwnership::Borrowed
            && (self.conversion.is_some()
                || self.transfer != ValueAdmissionTransfer::None
                || self.copies != ValueCopyAccounting::default())
        {
            return Err(invalid("borrowed value reports conversion or copy work"));
        }
        if let Some(conversion) = self.conversion {
            if self.source.numeric().map(|fact| fact.class) != Some(conversion.source)
                || self.target.numeric().map(|fact| fact.class) != Some(conversion.target)
            {
                return Err(invalid(
                    "numeric conversion differs from source or target facts",
                ));
            }
        }
        Ok(())
    }

    /// Validates the closed representation required for an immutable real64
    /// execution input. Borrowed runtime views are deliberately excluded: the
    /// receipt must describe storage whose lifetime is independent of the
    /// submitting runtime.
    pub fn validate_portable_immutable_real64(&self) -> Result<(), ContractError> {
        self.validate()?;
        if self.ownership == ValueAdmissionOwnership::Borrowed
            || self.target.kind
                != ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            || !matches!(
                self.target.storage,
                StorageFact::Scalar | StorageFact::Dense
            )
            || self.target.layout != LayoutFact::ColumnMajor
            || self.target.contiguity != ContiguityFact::Contiguous
            || self.target.residency != ResidencyFact::Host
            || self.target.alias != AliasFact::Unique
            || self.target.mutation != MutationFact::Immutable
            || self.target.certainty != CertaintyFact::Proven
        {
            return Err(invalid(
                "admitted value is not an owned portable immutable real64 array",
            ));
        }
        Ok(())
    }
}

fn invalid(reason: impl Into<String>) -> ContractError {
    ContractError::Invalid {
        field: "value_admission_receipt",
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use runmat_types::{ShapeFact, ViewFact};

    use super::*;

    fn portable_receipt(ownership: ValueAdmissionOwnership) -> ValueAdmissionReceipt {
        let mut target = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        );
        target.layout = LayoutFact::ColumnMajor;
        target.contiguity = ContiguityFact::Contiguous;
        target.view = ViewFact::Materialized;
        target.residency = ResidencyFact::Host;
        target.alias = AliasFact::Unique;
        target.mutation = MutationFact::Immutable;
        ValueAdmissionReceipt {
            schema_version: VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
            source: target.clone(),
            target,
            ownership,
            conversion: None,
            transfer: ValueAdmissionTransfer::None,
            copies: ValueCopyAccounting::default(),
        }
    }

    #[test]
    fn portable_immutable_real64_requires_independent_owned_storage() {
        portable_receipt(ValueAdmissionOwnership::Moved)
            .validate_portable_immutable_real64()
            .unwrap();

        let borrowed = portable_receipt(ValueAdmissionOwnership::Borrowed);
        assert!(borrowed.validate().is_ok());
        assert!(borrowed.validate_portable_immutable_real64().is_err());

        let mut device = portable_receipt(ValueAdmissionOwnership::Moved);
        device.target.residency = ResidencyFact::Device { provider: None };
        assert!(device.validate_portable_immutable_real64().is_err());
    }
}
