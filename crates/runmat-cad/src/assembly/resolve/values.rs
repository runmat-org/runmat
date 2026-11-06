//! Values in an assembly: lengths, angles, counts, vectors and offsets, selections,
//! configuration names, typed inputs and the overrides an assembly passes to its parts.

use super::*;

impl<'a, 'b> Resolver<'a, 'b> {
    pub(in crate::assembly) fn length(&self, e: &ValueExpr, env: &ValueEnv) -> Result<f64, String> {
        self.ev()
            .eval_as(e, env, Dimension::LENGTH)
            .map_err(|e| e.to_string())
    }

    pub(in crate::assembly) fn angle(&self, e: &ValueExpr, env: &ValueEnv) -> Result<f64, String> {
        self.ev()
            .eval_as(e, env, Dimension::ANGLE)
            .map_err(|e| e.to_string())
    }

    /// A dimensionless value (a ratio).
    pub(in crate::assembly) fn ratio(&self, e: &ValueExpr, env: &ValueEnv) -> Result<f64, String> {
        self.ev()
            .eval_as(e, env, Dimension::DIMENSIONLESS)
            .map_err(|e| e.to_string())
    }

    pub(in crate::assembly) fn count(&self, e: &ValueExpr, env: &ValueEnv) -> Result<u32, String> {
        let q = self.ev().eval(e, env).map_err(|e| e.to_string())?;
        if !q.dimension.is_dimensionless() || q.si < 1.0 || q.si.fract().abs() > 1e-9 {
            return Err(format!(
                "a count must be a whole number of at least 1, found {}",
                q.si
            ));
        }
        Ok(q.si.round() as u32)
    }

    pub(in crate::assembly) fn vec3(
        &self,
        v: &[ValueExpr; 3],
        env: &ValueEnv,
    ) -> Result<[f64; 3], String> {
        Ok([
            self.length(&v[0], env)?,
            self.length(&v[1], env)?,
            self.length(&v[2], env)?,
        ])
    }

    pub(in crate::assembly) fn offset(
        &self,
        o: &OffsetExpr,
        env: &ValueEnv,
    ) -> Result<Transform, String> {
        let t = self.vec3(&o.translation, env)?;
        let r = match &o.rotation {
            RotationExpr::Identity => Transform::IDENTITY,
            RotationExpr::AxisAngle { axis, angle } => {
                Transform::rotation_about([0.0; 3], *axis, self.angle(angle, env)?)
            }
            RotationExpr::Euler { yaw, pitch, roll } => {
                let z = Transform::rotation_about([0.0; 3], [0.0, 0.0, 1.0], self.angle(yaw, env)?);
                let y =
                    Transform::rotation_about([0.0; 3], [0.0, 1.0, 0.0], self.angle(pitch, env)?);
                let x =
                    Transform::rotation_about([0.0; 3], [1.0, 0.0, 0.0], self.angle(roll, env)?);
                z.compose(&y).compose(&x)
            }
            RotationExpr::Basis { x, z } => {
                let n = |a: [f64; 3]| {
                    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                    [a[0] / l, a[1] / l, a[2] / l]
                };
                let cr = |a: [f64; 3], b: [f64; 3]| {
                    [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ]
                };
                let zz = n(*z);
                let yy = n(cr(zz, *x));
                if !yy.iter().all(|v| v.is_finite()) {
                    return Err("the basis X and Z are parallel".into());
                }
                let xx = cr(yy, zz);
                Transform::from_frame(&crate::model::datum::Frame {
                    origin: [0.0; 3],
                    x: xx,
                    y: yy,
                    z: zz,
                })
            }
        };
        Ok(Transform::translation(t).compose(&r))
    }

    // ------------------------------------------------------------------ configurations

    pub(super) fn eval_selection(
        &mut self,
        sel: &ConfigurationSelection,
        env: &ValueEnv,
        summary: &DefinitionSummary,
        file: &FilePath,
        site: &Site,
    ) -> Result<ConfigKey, (DiagCode, String)> {
        match sel {
            ConfigurationSelection::Default => Ok(ConfigKey::default()),
            ConfigurationSelection::Named { name } => {
                self.check_config_name(name, summary)?;
                Ok(ConfigKey {
                    name: Some(name.clone()),
                    ..ConfigKey::default()
                })
            }
            ConfigurationSelection::Inputs { base, values } => {
                if let Some(b) = base {
                    self.check_config_name(b, summary)?;
                }
                let mut inputs = BTreeMap::new();
                let mut kinds = BTreeMap::new();
                for (name, expr) in values {
                    let si = self.typed_input(name, expr, env, summary)?;
                    inputs.insert(name.clone(), si);
                    if let Some((_, kind)) = summary.inputs.iter().find(|(n, _)| n == name) {
                        kinds.insert(name.clone(), kind.clone());
                    }
                }
                Ok(ConfigKey {
                    name: base.clone(),
                    inputs,
                    kinds,
                })
            }
            ConfigurationSelection::Switch { cases, otherwise } => {
                for c in cases {
                    if self.holds(Some(&c.when), file, site) {
                        return self.eval_selection(&c.select, env, summary, file, site);
                    }
                }
                self.eval_selection(otherwise, env, summary, file, site)
            }
        }
    }

    pub(super) fn check_config_name(
        &self,
        name: &ConfigurationName,
        s: &DefinitionSummary,
    ) -> Result<(), (DiagCode, String)> {
        if s.configurations.contains(name) {
            Ok(())
        } else {
            Err((
                DiagCode::AsmUnknownConfiguration,
                format!(
                    "`{}` has no configuration `{}` (it has: {})",
                    s.definition,
                    name,
                    s.configurations
                        .iter()
                        .map(|c| c.0.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ))
        }
    }

    pub(super) fn typed_input(
        &self,
        name: &InputName,
        expr: &ValueExpr,
        env: &ValueEnv,
        s: &DefinitionSummary,
    ) -> Result<f64, (DiagCode, String)> {
        let Some((_, kind)) = s.inputs.iter().find(|(n, _)| n == name) else {
            return Err((
                DiagCode::AsmInputType,
                format!("`{}` has no input `{}`", s.definition, name),
            ));
        };
        let Some(info) = self.ctx.units.kind(kind) else {
            return Err((
                DiagCode::AsmInputType,
                format!("input `{}` has an unknown kind `{}`", name, kind),
            ));
        };
        let q = self
            .ev()
            .eval(expr, env)
            .map_err(|e| (DiagCode::AsmInputType, format!("input `{}`: {}", name, e)))?;
        if q.dimension != info.dimension {
            return Err((
                DiagCode::AsmInputType,
                format!(
                    "input `{}` of `{}` is a {} ({}), found {}",
                    name, s.definition, info.id, info.si_symbol, q.dimension
                ),
            ));
        }
        Ok(q.si)
    }

    /// The overrides of a sub-assembly's inputs from its configuration key.
    pub(super) fn assembly_overrides(
        &self,
        doc: &AssemblyDocument,
        key: &ConfigKey,
    ) -> (BTreeMap<InputName, ValueExpr>, BTreeSet<String>) {
        let mut overrides = BTreeMap::new();
        let mut suppress = BTreeSet::new();
        if let Some(n) = &key.name {
            if let Some(c) = doc.configuration(n) {
                overrides = c.overrides.clone();
                suppress = c.suppress.clone();
            }
        }
        for (name, si) in &key.inputs {
            if let Some(decl) = doc.inputs.iter().find(|i| &i.name == name) {
                if let Some(info) = self.ctx.units.kind(&decl.kind) {
                    overrides.insert(name.clone(), ValueExpr::qty(*si, &info.si_symbol));
                }
            }
        }
        (overrides, suppress)
    }

    // ------------------------------------------------------------------ composition
}
