use std::collections::HashMap;
use std::sync::Once;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_geometry_core::{GeometryAsset, Region};
use runmat_macros::runtime_builtin;
use runmat_types::MemberAccess;
use runmat_value::{IntValue, ObjectInstance, Value};

use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::geometry_asset_from_value_with_builtin;

pub const GEOMETRY_REGION_CLASS: &str = "geometry.Region";
pub const GEOMETRY_REGION_NAME: &str = "geometry.region";
const GEOMETRY_REGION_TYPE_IDENTITY: &str = "runmat.geometry.region";
const GEOMETRY_REGION_FIELDS: &[&str] = &[
    "geometry_id",
    "geometry_revision",
    "region_id",
    "name",
    "tag",
];
const GEOMETRY_REGION_REQUIRED_FIELDS: &[&str] =
    &["geometry_id", "geometry_revision", "region_id", "name"];

const REGION_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "asset",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Loaded geometry.Asset object.",
    },
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact region ID, name, or tag; optionally prefixed with id:, name:, or tag:.",
    },
];
const REGION_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "region",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Resolved immutable geometry.Region value.",
}];
const REGION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "region = geometry.region(asset, selector)",
    inputs: &REGION_INPUTS,
    outputs: &REGION_OUTPUTS,
}];
const REGION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.REGION.INVALID",
    identifier: Some("RunMat:geometry:region:InvalidRegion"),
    when: "The asset is invalid, or the selector is empty, missing, or ambiguous.",
    message: "geometry.region: invalid region selection",
};
const REGION_ERRORS: [BuiltinErrorDescriptor; 1] = [REGION_ERROR];

pub const GEOMETRY_REGION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REGION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REGION_ERRORS,
};

pub const GEOMETRY_REGION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "geometry.region resolves text metadata from a geometry.Asset; its revision is returned as a native uint32 value and it has no integer input role.",
    };

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeometryRegionRef {
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub region_id: String,
    pub name: String,
    pub tag: Option<String>,
}

#[runtime_builtin(
    name = "geometry.region",
    category = "geometry",
    summary = "Resolve one region from a geometry asset by exact ID, name, or tag.",
    keywords = "geometry,region,selector,cad,fea",
    descriptor(crate::builtins::geometry::region::GEOMETRY_REGION_DESCRIPTOR),
    integer_audit(crate::builtins::geometry::region::GEOMETRY_REGION_INTEGER_AUDIT),
    builtin_path = "crate::builtins::geometry::region"
)]
pub async fn geometry_region_builtin(asset: Value, selector: String) -> BuiltinResult<Value> {
    let asset = geometry_asset_from_value_with_builtin(&asset, GEOMETRY_REGION_NAME)?;
    let region = resolve_region(&asset, &selector)?;
    region_to_value(&asset, region)
}

pub(crate) fn region_ref_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<GeometryRegionRef> {
    let Value::Object(object) = value else {
        return Err(region_error(
            builtin,
            format!("expected {GEOMETRY_REGION_CLASS}"),
        ));
    };
    if object.class_name != GEOMETRY_REGION_CLASS {
        return Err(region_error(
            builtin,
            format!(
                "expected {GEOMETRY_REGION_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    validate_region_object(object).map_err(|message| region_error(builtin, message))?;
    let string = |name: &str| match object.properties.get(name) {
        Some(Value::String(value)) => Ok(value.clone()),
        _ => Err(region_error(
            builtin,
            format!("{GEOMETRY_REGION_CLASS}.{name} must be a string"),
        )),
    };
    let geometry_revision = match object.properties.get("geometry_revision") {
        Some(Value::Int(IntValue::U32(value))) => *value,
        _ => {
            return Err(region_error(
                builtin,
                format!("{GEOMETRY_REGION_CLASS}.geometry_revision must be uint32"),
            ))
        }
    };
    let tag = match object.properties.get("tag") {
        Some(Value::String(value)) => Some(value.clone()),
        None => None,
        _ => {
            return Err(region_error(
                builtin,
                format!("{GEOMETRY_REGION_CLASS}.tag must be a string when present"),
            ))
        }
    };
    Ok(GeometryRegionRef {
        geometry_id: string("geometry_id")?,
        geometry_revision,
        region_id: string("region_id")?,
        name: string("name")?,
        tag,
    })
}

fn resolve_region<'a>(asset: &'a GeometryAsset, selector: &str) -> BuiltinResult<&'a Region> {
    if selector.is_empty() {
        return Err(region_error(
            GEOMETRY_REGION_NAME,
            "selector must not be empty",
        ));
    }
    let exact_region_id_exists = asset
        .regions
        .iter()
        .any(|region| region.region_id == selector);
    let (mode, value) = if exact_region_id_exists {
        (Some("id"), selector)
    } else {
        match selector.split_once(':') {
            Some((prefix @ ("id" | "region" | "name" | "tag"), value)) => (Some(prefix), value),
            Some((prefix, _)) => {
                return Err(region_error(
                    GEOMETRY_REGION_NAME,
                    format!("unsupported selector prefix `{prefix}`"),
                ))
            }
            None => (None, selector),
        }
    };
    if value.is_empty() {
        return Err(region_error(
            GEOMETRY_REGION_NAME,
            "selector value must not be empty",
        ));
    }
    let matches = asset
        .regions
        .iter()
        .filter(|region| match mode {
            Some("id") | Some("region") => region.region_id == value,
            Some("name") => region.name == value,
            Some("tag") => region.tag.as_deref() == Some(value),
            Some(_) => unreachable!("selector modes are validated before matching"),
            None => {
                region.region_id == value
                    || region.name == value
                    || region.tag.as_deref() == Some(value)
            }
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [region] => Ok(*region),
        [] => Err(region_error(
            GEOMETRY_REGION_NAME,
            format!("selector `{selector}` did not match any region"),
        )),
        _ => Err(region_error(
            GEOMETRY_REGION_NAME,
            format!(
                "selector `{selector}` matched {} regions; use an exact id: selector",
                matches.len()
            ),
        )),
    }
}

fn region_to_value(asset: &GeometryAsset, region: &Region) -> BuiltinResult<Value> {
    ensure_region_class_registered();
    let mut object = ObjectInstance::new(GEOMETRY_REGION_CLASS.to_string());
    object.properties.insert(
        "geometry_id".into(),
        Value::String(asset.geometry_id.clone()),
    );
    object.properties.insert(
        "geometry_revision".into(),
        Value::Int(IntValue::U32(asset.revision)),
    );
    object
        .properties
        .insert("region_id".into(), Value::String(region.region_id.clone()));
    object
        .properties
        .insert("name".into(), Value::String(region.name.clone()));
    if let Some(tag) = &region.tag {
        object
            .properties
            .insert("tag".into(), Value::String(tag.clone()));
    }
    Ok(Value::Object(object))
}

pub(crate) fn ensure_region_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    PORTABLE_SCHEMA.call_once(|| {
        register_immutable_value_class(ImmutableValueClassSchema {
            class_name: GEOMETRY_REGION_CLASS,
            type_identity: GEOMETRY_REGION_TYPE_IDENTITY,
            schema_version: 1,
            allowed_fields: GEOMETRY_REGION_FIELDS,
            required_fields: GEOMETRY_REGION_REQUIRED_FIELDS,
            validate: validate_region_object,
        })
        .unwrap_or_else(|error| panic!("failed to register {GEOMETRY_REGION_CLASS}: {error}"));
    });
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(GEOMETRY_REGION_CLASS);
    RUNTIME_CLASS.ensure(|| {
        let properties = GEOMETRY_REGION_FIELDS
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    crate::class_registry::RuntimeProperty {
                        name: (*name).to_string(),
                        is_static: false,
                        is_constant: true,
                        is_dependent: false,
                        get_access: MemberAccess::Public,
                        set_access: MemberAccess::Private,
                        default_value: None,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: GEOMETRY_REGION_CLASS.to_string(),
            parent: None,
            properties,
            methods: HashMap::new(),
        });
    });
}

fn validate_region_object(object: &ObjectInstance) -> Result<(), String> {
    for name in ["geometry_id", "region_id", "name"] {
        match object.properties.get(name) {
            Some(Value::String(value)) if !value.is_empty() => {}
            Some(Value::String(_)) => {
                return Err(format!("{GEOMETRY_REGION_CLASS}.{name} must not be empty"))
            }
            _ => return Err(format!("{GEOMETRY_REGION_CLASS}.{name} must be a string")),
        }
    }
    if !matches!(
        object.properties.get("geometry_revision"),
        Some(Value::Int(IntValue::U32(_)))
    ) {
        return Err(format!(
            "{GEOMETRY_REGION_CLASS}.geometry_revision must be uint32"
        ));
    }
    if let Some(tag) = object.properties.get("tag") {
        if !matches!(tag, Value::String(value) if !value.is_empty()) {
            return Err(format!(
                "{GEOMETRY_REGION_CLASS}.tag must be a non-empty string when present"
            ));
        }
    }
    Ok(())
}

fn region_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(format!("{}: {}", REGION_ERROR.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(
            REGION_ERROR
                .identifier
                .unwrap_or("RunMat:geometry:region:InvalidRegion"),
        )
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::geometry::geometry_load_builtin;
    use crate::execution::value_codec::{decode_inline_value, encode_inline_value};
    use futures::executor::block_on;

    fn asset() -> GeometryAsset {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("fixture.stl");
        std::fs::write(
            &path,
            "solid fixture\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid fixture\n",
        )
        .unwrap();
        let value = block_on(geometry_load_builtin(vec![Value::String(
            path.to_string_lossy().to_string(),
        )]))
        .expect("fixture geometry should load");
        let mut asset = geometry_asset_from_value_with_builtin(&value, GEOMETRY_REGION_NAME)
            .expect("fixture asset should decode");
        asset.geometry_id = "fixture".into();
        asset.revision = 7;
        asset.regions = vec![
            Region {
                region_id: "face:1".into(),
                name: "Base".into(),
                tag: Some("support".into()),
                cad_ownership: None,
            },
            Region {
                region_id: "face:2".into(),
                name: "Load".into(),
                tag: Some("pressure".into()),
                cad_ownership: None,
            },
        ];
        asset
    }

    #[test]
    fn resolves_id_name_and_tag_to_a_revision_bound_value() {
        let asset = asset();
        for selector in ["face:1", "id:face:1", "name:Base", "tag:support"] {
            let region = resolve_region(&asset, selector).expect("region should resolve");
            assert_eq!(region.region_id, "face:1");
        }
        let value = region_to_value(&asset, &asset.regions[0]).expect("region value");
        let reference = region_ref_from_value(GEOMETRY_REGION_NAME, &value).expect("region ref");
        assert_eq!(reference.geometry_id, "fixture");
        assert_eq!(reference.geometry_revision, 7);
        assert_eq!(reference.region_id, "face:1");
        assert_eq!(reference.name, "Base");
        assert_eq!(reference.tag.as_deref(), Some("support"));
        assert_eq!(
            decode_inline_value(&encode_inline_value(&value).expect("encode")).expect("decode"),
            value
        );
    }

    #[test]
    fn rejects_missing_ambiguous_and_unknown_prefixed_selectors() {
        let mut asset = asset();
        asset.regions[1].name = "Base".into();
        assert!(resolve_region(&asset, "Base")
            .unwrap_err()
            .message()
            .contains("matched 2 regions"));
        assert!(resolve_region(&asset, "missing").is_err());
        assert!(resolve_region(&asset, "label:Base").is_err());
    }
}
