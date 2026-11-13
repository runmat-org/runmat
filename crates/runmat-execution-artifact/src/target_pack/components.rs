use super::*;
use crate::{ObjectDescriptor, ObjectNamespace};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Bounded artifact label, never a semantic operation or service identity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ComponentId(String);
impl ComponentId {
    pub fn new(value: impl Into<String>) -> ArtifactResult<Self> {
        let value = value.into();
        if value.len() > 256 {
            return Err(invalid("target pack component label is too long"));
        }
        crate::object::validate_logical_name(&value)?;
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for ComponentId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetPackComponentKind {
    NativeAbiProbe,
    NativeObject,
    StaticArchive,
    RuntimeData,
    Toolchain,
    Sysroot,
    NumericalEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPackComponent {
    pub id: ComponentId,
    pub kind: TargetPackComponentKind,
    pub object: ObjectDescriptor,
    pub dependencies: Vec<ComponentId>,
    /// Requirements of retaining this runtime component, not provided services.
    pub capabilities: runmat_types::CapabilitySet,
    pub effects: runmat_types::EffectSet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeLinkRetention {
    Referenced,
    WholeArchive,
}

/// Ordered native inputs. Repeated archives are intentional and preserved;
/// mutual component dependencies are not incorrectly treated as a DAG.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NativeLinkInput {
    Component {
        component: ComponentId,
        retention: NativeLinkRetention,
    },
    HostedLibrary {
        required_by: ComponentId,
        name: String,
    },
    HostedFramework {
        required_by: ComponentId,
        name: String,
    },
}
impl NativeLinkInput {
    pub(super) fn owner(&self) -> &ComponentId {
        match self {
            Self::Component { component, .. } => component,
            Self::HostedLibrary { required_by, .. } | Self::HostedFramework { required_by, .. } => {
                required_by
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NativeBuildInput {
    ContentAddressed {
        component: ComponentId,
    },
    /// A truthful non-hermetic identity; never upgraded to content evidence.
    HostedDeclared {
        fingerprint: Digest,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBuildInputs {
    pub linker: NativeBuildInput,
    pub sysroot: NativeBuildInput,
}

pub(super) fn index(
    components: &[TargetPackComponent],
) -> ArtifactResult<BTreeMap<ComponentId, &TargetPackComponent>> {
    if components.is_empty()
        || components.len() > MAX_COMPONENTS
        || components.windows(2).any(|pair| pair[0].id >= pair[1].id)
    {
        return Err(invalid(
            "target pack component inventory is not bounded, sorted and unique",
        ));
    }
    let mut edges = 0usize;
    let mut bytes = 0u64;
    let index = components
        .iter()
        .map(|component| (component.id.clone(), component))
        .collect::<BTreeMap<_, _>>();
    for component in components {
        component.object.validate()?;
        nonzero(component.object.digest)?;
        bytes = bytes
            .checked_add(component.object.encoded_length)
            .ok_or_else(|| invalid("target pack component size overflows"))?;
        if component.object.namespace != ObjectNamespace::ProgramArtifact
            || component.object.encoded_length == 0
            || component.object.encoded_length > MAX_COMPONENT_BYTES
            || bytes > MAX_PACK_BYTES
            || component
                .dependencies
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "target pack component descriptor/dependencies are invalid",
            ));
        }
        edges = edges
            .checked_add(component.dependencies.len())
            .ok_or_else(|| invalid("target pack dependency count overflows"))?;
        if edges > MAX_EDGES
            || component
                .dependencies
                .iter()
                .any(|id| !index.contains_key(id))
        {
            return Err(invalid(
                "target pack dependency inventory is missing or exceeds its bound",
            ));
        }
    }
    Ok(index)
}

/// SCC-safe exact closure, without changing declared native link order.
pub(super) fn retain(
    index: &BTreeMap<ComponentId, &TargetPackComponent>,
    roots: impl IntoIterator<Item = ComponentId>,
    reached: &mut BTreeSet<ComponentId>,
) -> ArtifactResult<()> {
    let mut pending = roots.into_iter().collect::<Vec<_>>();
    while let Some(id) = pending.pop() {
        let component = index
            .get(&id)
            .ok_or_else(|| invalid("required target pack component is missing"))?;
        if reached.insert(id) {
            pending.extend(component.dependencies.iter().cloned());
        }
    }
    Ok(())
}
