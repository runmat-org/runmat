use std::collections::BTreeSet;
use std::str::FromStr;

use anyhow::{Context, Result};
use runmat_execution::resource::{
    AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
    AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId, AcceleratorProviderVersion,
    AcceleratorRequest,
};
use runmat_execution::Digest;

use crate::public_api::types;

pub fn accelerator_device_to_api(value: AcceleratorDevice) -> Result<types::AcceleratorDeviceBody> {
    value.validate().context("invalid accelerator device")?;
    Ok(types::AcceleratorDeviceBody {
        id: value.id.to_string().into(),
        allocation_domain: value.allocation_domain.to_string().into(),
        class: value.class.to_string().into(),
        provider: provider_to_api(value.provider),
        max_allocation_bytes: to_i64(value.max_allocation_bytes, "accelerator maximum allocation")?,
        features: value.features.into_iter().map(feature_to_api).collect(),
        inventory_epoch: to_i64(value.inventory_epoch, "accelerator inventory epoch")?,
    })
}

pub fn accelerator_device_from_api(
    value: types::AcceleratorDeviceBody,
) -> Result<AcceleratorDevice> {
    let device = AcceleratorDevice {
        id: AcceleratorDeviceId::new(String::from(value.id))?,
        allocation_domain: AcceleratorAllocationDomainId::new(String::from(
            value.allocation_domain,
        ))?,
        class: AcceleratorClass::new(String::from(value.class))?,
        provider: provider_from_api(value.provider)?,
        max_allocation_bytes: to_u64(value.max_allocation_bytes, "accelerator maximum allocation")?,
        features: value
            .features
            .into_iter()
            .map(feature_from_api)
            .collect::<Result<BTreeSet<_>>>()?,
        inventory_epoch: to_u64(value.inventory_epoch, "accelerator inventory epoch")?,
    };
    device.validate().context("invalid accelerator device")?;
    Ok(device)
}

pub fn accelerator_request_to_api(
    value: AcceleratorRequest,
) -> Result<types::AcceleratorRequestBody> {
    value.validate().context("invalid accelerator request")?;
    Ok(types::AcceleratorRequestBody {
        class: value.class.to_string().into(),
        count: i32::from(value.count),
        minimum_allocation_bytes: to_i64(
            value.minimum_allocation_bytes,
            "accelerator minimum allocation",
        )?,
        provider: value.provider.map(provider_to_api),
        required_features: value
            .required_features
            .into_iter()
            .map(feature_to_api)
            .collect(),
    })
}

pub fn accelerator_request_from_api(
    value: types::AcceleratorRequestBody,
) -> Result<AcceleratorRequest> {
    let count = u16::try_from(value.count).context("accelerator request count is out of range")?;
    if count == 0 {
        anyhow::bail!("accelerator request count must be non-zero");
    }
    let request = AcceleratorRequest {
        class: AcceleratorClass::new(String::from(value.class))?,
        count,
        minimum_allocation_bytes: to_u64(
            value.minimum_allocation_bytes,
            "accelerator minimum allocation",
        )?,
        provider: value.provider.map(provider_from_api).transpose()?,
        required_features: value
            .required_features
            .into_iter()
            .map(feature_from_api)
            .collect::<Result<BTreeSet<_>>>()?,
    };
    request.validate().context("invalid accelerator request")?;
    Ok(request)
}

fn provider_to_api(value: AcceleratorProvider) -> types::AcceleratorProviderBody {
    types::AcceleratorProviderBody {
        id: value.id.to_string().into(),
        version: value.version.to_string().into(),
        abi_fingerprint: value.abi_fingerprint.to_string().into(),
    }
}

fn provider_from_api(value: types::AcceleratorProviderBody) -> Result<AcceleratorProvider> {
    Ok(AcceleratorProvider {
        id: AcceleratorProviderId::new(String::from(value.id))?,
        version: AcceleratorProviderVersion::new(String::from(value.version))?,
        abi_fingerprint: Digest::from_str(&String::from(value.abi_fingerprint))?,
    })
}

fn feature_to_api(value: AcceleratorFeature) -> types::AcceleratorFeatureBody {
    match value {
        AcceleratorFeature::Compute => types::AcceleratorFeatureBody::Compute,
        AcceleratorFeature::UnifiedMemory => types::AcceleratorFeatureBody::UnifiedMemory,
        AcceleratorFeature::PeerToPeer => types::AcceleratorFeatureBody::PeerToPeer,
        AcceleratorFeature::ConfidentialCompute => {
            types::AcceleratorFeatureBody::ConfidentialCompute
        }
        AcceleratorFeature::Custom(value) => types::AcceleratorFeatureBody::Custom(value),
    }
}

fn feature_from_api(value: types::AcceleratorFeatureBody) -> Result<AcceleratorFeature> {
    Ok(match value {
        types::AcceleratorFeatureBody::Compute => AcceleratorFeature::Compute,
        types::AcceleratorFeatureBody::UnifiedMemory => AcceleratorFeature::UnifiedMemory,
        types::AcceleratorFeatureBody::PeerToPeer => AcceleratorFeature::PeerToPeer,
        types::AcceleratorFeatureBody::ConfidentialCompute => {
            AcceleratorFeature::ConfidentialCompute
        }
        types::AcceleratorFeatureBody::Custom(value) => {
            if value.is_empty()
                || value.len() > 128
                || !value.is_ascii()
                || value.chars().any(char::is_control)
            {
                anyhow::bail!("custom accelerator feature must be 1..=128 printable ASCII bytes");
            }
            AcceleratorFeature::Custom(value)
        }
    })
}

fn to_i64(value: u64, field: &str) -> Result<i64> {
    i64::try_from(value).with_context(|| format!("{field} exceeds the public API range"))
}

fn to_u64(value: i64, field: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{field} must be non-negative"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_round_trip_preserves_provider_view_and_physical_domain() {
        let device = AcceleratorDevice {
            id: AcceleratorDeviceId::new("device:wgpu-primary").unwrap(),
            allocation_domain: AcceleratorAllocationDomainId::new("allocation:physical-primary")
                .unwrap(),
            class: AcceleratorClass::new("gpu").unwrap(),
            provider: AcceleratorProvider {
                id: AcceleratorProviderId::new("runmat.wgpu").unwrap(),
                version: AcceleratorProviderVersion::new("0.6.2").unwrap(),
                abi_fingerprint: Digest::sha256(b"wgpu-provider-abi"),
            },
            max_allocation_bytes: 1 << 30,
            features: BTreeSet::from([AcceleratorFeature::Compute]),
            inventory_epoch: 7,
        };

        let api = accelerator_device_to_api(device.clone()).unwrap();
        assert_eq!(accelerator_device_from_api(api).unwrap(), device);
    }
}
