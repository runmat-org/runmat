//! The seams a fastener set reads through: the fastener catalog (parts by table row) and the
//! stack gauge (member thicknesses).

use super::*;

// ------------------------------------------------------------------------------ seams

/// A selected fastener: the family configuration (dash number) for a diameter and grip.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FastenerPart {
    pub family: DefinitionRef,
    pub configuration: ConfigurationName,
    pub part_number: String,
}

pub trait FastenerCatalog {
    fn select(
        &self,
        family: &DefinitionRef,
        diameter: f64,
        grip: f64,
    ) -> Result<FastenerPart, String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogRow {
    pub family: DefinitionRef,
    pub diameter: f64,
    pub grip_min: f64,
    pub grip_max: f64,
    pub configuration: ConfigurationName,
    pub part_number: String,
}

/// A dash-number table (a family's configurations from table).
#[derive(Clone, Debug, Default)]
pub struct TableCatalog {
    pub rows: Vec<CatalogRow>,
}

impl FastenerCatalog for TableCatalog {
    fn select(
        &self,
        family: &DefinitionRef,
        diameter: f64,
        grip: f64,
    ) -> Result<FastenerPart, String> {
        let tol = 1e-9 * diameter.abs().max(1e-3);
        let mut best: Option<&CatalogRow> = None;
        for r in &self.rows {
            if &r.family != family || (r.diameter - diameter).abs() > tol {
                continue;
            }
            if grip < r.grip_min - 1e-12 || grip > r.grip_max + 1e-12 {
                continue;
            }
            best = match best {
                Some(b) if (b.grip_max, &b.part_number) <= (r.grip_max, &r.part_number) => Some(b),
                _ => Some(r),
            };
        }
        best.map(|r| FastenerPart {
            family: r.family.clone(),
            configuration: r.configuration.clone(),
            part_number: r.part_number.clone(),
        })
        .ok_or_else(|| {
            format!(
                "`{}` has no size for diameter {} m and grip {} m",
                family, diameter, grip
            )
        })
    }
}

/// A stack member at expansion time: its path, its definition and where it sits.
#[derive(Clone, Debug)]
pub struct StackMember {
    pub path: RelPath,
    pub definition: DefinitionRef,
    /// Member definition → assembly coordinates.
    pub transform: Transform,
    pub summary: Option<Arc<ConfiguredSummary>>,
}

/// Measures the clamped stack at a point along an axis (the kernel ray-casts; R0 uses
/// published nominal thicknesses).
pub trait StackGauge {
    fn grip(&self, members: &[StackMember], point: [f64; 3], axis: [f64; 3])
        -> Result<f64, String>;
}

/// The sum of each member's published value `thickness` (or another name).
#[derive(Clone, Debug)]
pub struct NominalThicknessGauge {
    pub publish: PublishName,
}

impl Default for NominalThicknessGauge {
    fn default() -> Self {
        Self {
            publish: PublishName::new("thickness"),
        }
    }
}

impl StackGauge for NominalThicknessGauge {
    fn grip(
        &self,
        members: &[StackMember],
        _point: [f64; 3],
        _axis: [f64; 3],
    ) -> Result<f64, String> {
        let mut total = 0.0;
        for m in members {
            let s = m
                .summary
                .as_ref()
                .ok_or_else(|| format!("`{}` has no configured summary", m.path))?;
            match s.publishes.get(&self.publish) {
                Some(PublishedGeometry::Value { value })
                    if value.dimension == Dimension::LENGTH =>
                {
                    total += value.si
                }
                _ => {
                    return Err(format!(
                        "`{}` publishes no length `{}`",
                        m.definition, self.publish
                    ))
                }
            }
        }
        Ok(total)
    }
}
