use super::*;
use crate::builtins::fea::values::{quantity_scalar, TEMPERATURE_CLASS};
use crate::builtins::geometry::region::{region_ref_from_value, GeometryRegionRef};

pub(super) const STRUCTURAL_FIXED_NAME: &str = "fea.StructuralConstraint.fixed";
pub(super) const THERMAL_PRESCRIBED_NAME: &str = "fea.ThermalConstraint.prescribed";

const FIXED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Constraint identifier.",
    },
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A geometry.Region selected from the model geometry.",
    },
];
const FIXED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.fixed(id, region)",
    inputs: &FIXED_INPUTS,
    outputs: &OUT_ANY,
}];

pub(super) const STRUCTURAL_FIXED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIXED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub(super) const STRUCTURAL_FIXED_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "The fixed structural-constraint constructor accepts an identifier and an immutable geometry.Region; it has no integer input role.",
    };

const THERMAL_PRESCRIBED_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Constraint identifier.",
    },
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A geometry.Region selected from the model geometry.",
    },
    BuiltinParamDescriptor {
        name: "temperature",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A fea.Temperature value.",
    },
];
const THERMAL_PRESCRIBED_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.ThermalConstraint.prescribed(id, region, temperature)",
        inputs: &THERMAL_PRESCRIBED_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(super) const THERMAL_PRESCRIBED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &THERMAL_PRESCRIBED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub(super) const THERMAL_PRESCRIBED_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "The prescribed-temperature constructor accepts an identifier, geometry.Region, and already-dimensioned fea.Temperature value; it has no direct integer input role.",
    };

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundConstraint<T> {
    pub(super) geometry_id: String,
    pub(super) geometry_revision: u32,
    pub(super) constraint: T,
}

pub(super) type BoundStructuralConstraint = GeometryBoundConstraint<StructuralConstraint>;
pub(super) type BoundThermalConstraint = GeometryBoundConstraint<ThermalConstraint>;

impl<T> GeometryBoundConstraint<T> {
    pub(super) fn require_geometry(
        &self,
        builtin: &'static str,
        geometry: &GeometryAsset,
    ) -> BuiltinResult<()> {
        if self.geometry_id != geometry.geometry_id || self.geometry_revision != geometry.revision {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!(
                    "{} belongs to geometry `{}` revision {}, but the model uses `{}` revision {}",
                    FEA_STRUCTURAL_CONSTRAINT_CLASS,
                    self.geometry_id,
                    self.geometry_revision,
                    geometry.geometry_id,
                    geometry.revision
                ),
            ));
        }
        Ok(())
    }
}

#[runtime_builtin(
    name = "fea.StructuralConstraint.fixed",
    category = "fea",
    summary = "Create a fixed structural constraint on a selected geometry region.",
    keywords = "fea,structural,constraint,fixed,region",
    descriptor(crate::builtins::fea::constraints::STRUCTURAL_FIXED_DESCRIPTOR),
    integer_audit(crate::builtins::fea::constraints::STRUCTURAL_FIXED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints"
)]
pub async fn structural_fixed_builtin(id: String, region: Value) -> BuiltinResult<Value> {
    if id.is_empty() {
        return Err(builtin_error(
            STRUCTURAL_FIXED_NAME,
            &ERROR_INPUT,
            "constraint id must not be empty",
        ));
    }
    let GeometryRegionRef {
        geometry_id,
        geometry_revision,
        region_id,
        ..
    } = region_ref_from_value(STRUCTURAL_FIXED_NAME, &region)?;
    let value = GeometryBoundConstraint {
        geometry_id,
        geometry_revision,
        constraint: StructuralConstraint {
            constraint_id: id,
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Fixed {
                target: RegionTarget::boundary(region_id),
            },
        },
    };
    serializable_to_object_preserving_integers(
        STRUCTURAL_FIXED_NAME,
        &ERROR_INTERNAL,
        FEA_STRUCTURAL_CONSTRAINT_CLASS,
        &value,
        &[],
        &["geometry_revision"],
    )
}

#[runtime_builtin(
    name = "fea.ThermalConstraint.prescribed",
    category = "fea",
    summary = "Create a prescribed-temperature constraint on a selected geometry region.",
    keywords = "fea,thermal,constraint,temperature,region",
    descriptor(crate::builtins::fea::constraints::THERMAL_PRESCRIBED_DESCRIPTOR),
    integer_audit(crate::builtins::fea::constraints::THERMAL_PRESCRIBED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints"
)]
pub async fn thermal_prescribed_builtin(
    id: String,
    region: Value,
    temperature: Value,
) -> BuiltinResult<Value> {
    if id.is_empty() {
        return Err(builtin_error(
            THERMAL_PRESCRIBED_NAME,
            &ERROR_INPUT,
            "constraint id must not be empty",
        ));
    }
    let GeometryRegionRef {
        geometry_id,
        geometry_revision,
        region_id,
        ..
    } = region_ref_from_value(THERMAL_PRESCRIBED_NAME, &region)?;
    let kelvin = quantity_scalar(
        THERMAL_PRESCRIBED_NAME,
        &temperature,
        TEMPERATURE_CLASS,
        "kelvin",
    )?;
    let value = GeometryBoundConstraint {
        geometry_id,
        geometry_revision,
        constraint: ThermalConstraint {
            constraint_id: id,
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(region_id),
                value: PrescribedValue::Constant(Temperature::new(kelvin)),
            },
        },
    };
    serializable_to_object_preserving_integers(
        THERMAL_PRESCRIBED_NAME,
        &ERROR_INTERNAL,
        FEA_THERMAL_CONSTRAINT_CLASS,
        &value,
        &[],
        &["geometry_revision"],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::geometry::region::geometry_region_builtin;
    use crate::builtins::geometry::{
        geometry_asset_from_value_with_builtin, geometry_load_builtin,
    };
    use futures::executor::block_on;

    fn load_fixture(path: &std::path::Path, x: f64) -> Value {
        std::fs::write(
            path,
            format!(
                "solid part\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex {x} 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid part\n"
            ),
        )
        .unwrap();
        block_on(geometry_load_builtin(vec![Value::String(
            path.to_string_lossy().to_string(),
        )]))
        .expect("geometry should load")
    }

    #[test]
    fn fixed_constraint_is_bound_to_the_selected_geometry_revision() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.stl");
        std::fs::write(
            &path,
            "solid base\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid base\n",
        )
        .unwrap();
        let asset = block_on(geometry_load_builtin(vec![Value::String(
            path.to_string_lossy().to_string(),
        )]))
        .expect("geometry should load");
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let region_id = decoded.regions[0].region_id.clone();
        let region = block_on(geometry_region_builtin(asset, region_id)).expect("region");
        let constraint =
            block_on(structural_fixed_builtin("base".into(), region)).expect("fixed constraint");
        let bound: BoundStructuralConstraint = object_payload(
            STRUCTURAL_FIXED_NAME,
            &constraint,
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
        )
        .expect("constraint should decode");
        assert_eq!(bound.geometry_id, decoded.geometry_id);
        assert_eq!(bound.geometry_revision, decoded.revision);
        assert!(matches!(
            bound.constraint.kind,
            StructuralConstraintKind::Fixed { ref target } if target.region_id == decoded.regions[0].region_id
        ));

        let mut changed = decoded;
        changed.revision += 1;
        assert!(bound
            .require_geometry(STRUCTURAL_FIXED_NAME, &changed)
            .unwrap_err()
            .message()
            .contains("belongs to geometry"));
    }

    #[test]
    fn model_composition_accepts_matching_geometry_and_rejects_a_foreign_region() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_fixture(&tmp.path().join("part-a.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let region = block_on(geometry_region_builtin(
            asset.clone(),
            decoded.regions[0].region_id.clone(),
        ))
        .expect("region");
        let constraint =
            block_on(structural_fixed_builtin("base".into(), region)).expect("fixed constraint");
        let model = create_model_object_from_args(vec![
            Value::String("matching".into()),
            asset,
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("BoundaryConditions".into()),
            constraint.clone(),
        ])
        .expect("matching geometry should compose");
        let model: AnalysisModel =
            object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).expect("model should decode");
        assert_eq!(model.constraints.len(), 1);

        let foreign_asset = load_fixture(&tmp.path().join("part-b.stl"), 2.0);
        let error = create_model_object_from_args(vec![
            Value::String("foreign".into()),
            foreign_asset,
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("BoundaryConditions".into()),
            constraint,
        ])
        .unwrap_err();
        assert!(error.message().contains("belongs to geometry"));
    }

    #[test]
    fn thermal_constraint_requires_a_typed_temperature() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_fixture(&tmp.path().join("thermal.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, THERMAL_PRESCRIBED_NAME)
            .expect("asset should decode");
        let region = block_on(geometry_region_builtin(
            asset,
            decoded.regions[0].region_id.clone(),
        ))
        .expect("region");
        let temperature = block_on(crate::builtins::fea::values::temperature_celsius_builtin(
            Value::Num(20.0),
        ))
        .expect("temperature");
        let constraint = block_on(thermal_prescribed_builtin(
            "wall_temperature".into(),
            region.clone(),
            temperature,
        ))
        .expect("thermal constraint");
        let bound: BoundThermalConstraint = object_payload(
            THERMAL_PRESCRIBED_NAME,
            &constraint,
            FEA_THERMAL_CONSTRAINT_CLASS,
        )
        .expect("constraint should decode");
        assert!(matches!(
            bound.constraint.kind,
            ThermalConstraintKind::Prescribed {
                value: PrescribedValue::Constant(Temperature { kelvin }),
                ..
            } if kelvin == 293.15
        ));

        let displacement = block_on(crate::builtins::fea::values::displacement_meters_builtin(
            Value::Num(1.0),
        ))
        .expect("displacement");
        assert!(block_on(thermal_prescribed_builtin(
            "wrong_dimension".into(),
            region,
            displacement,
        ))
        .is_err());
    }
}
