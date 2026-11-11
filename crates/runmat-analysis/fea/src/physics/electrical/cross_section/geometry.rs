//! The cross-section as the mesher and the operator see it: y up, the solve domain between the
//! reference planes that bound the signal layer (or open air where an outer layer has none),
//! dielectric bands, the trapezoidal traces and the conformal solder mask.
use super::{
    ConductorRole, CopperRole, CrossSection, CrossSectionError, Enclosure, StackLayerKind,
    TraceBase,
};

/// Enclosure distances in units of the dielectric height between the signal layer and its
/// farthest bounding plane (or its single plane). At these distances the side walls and the
/// lid change a microstrip's Z0 by less than 0.05 % (see the enclosure test).
pub(super) const ENCLOSURE_HEIGHTS: f64 = 60.0;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Trace {
    pub name: String,
    pub role: ConductorRole,
    /// Base-face corners and far-face corners (x), and the base and far y.
    pub base_left: f64,
    pub base_right: f64,
    pub far_left: f64,
    pub far_right: f64,
    pub base_y: f64,
    pub far_y: f64,
}

impl Trace {
    pub fn y_range(&self) -> (f64, f64) {
        (self.base_y.min(self.far_y), self.base_y.max(self.far_y))
    }

    /// Left and right edge x at height `y` inside the trace's band.
    pub fn edges_at(&self, y: f64) -> (f64, f64) {
        let span = self.far_y - self.base_y;
        let f = if span == 0.0 {
            0.0
        } else {
            ((y - self.base_y) / span).clamp(0.0, 1.0)
        };
        (
            self.base_left + f * (self.far_left - self.base_left),
            self.base_right + f * (self.far_right - self.base_right),
        )
    }

    pub fn contains(&self, x: f64, y: f64, tolerance: f64) -> bool {
        let (lo, hi) = self.y_range();
        if y < lo - tolerance || y > hi + tolerance {
            return false;
        }
        let (left, right) = self.edges_at(y);
        x >= left - tolerance && x <= right + tolerance
    }
}

/// Conformal solder mask over the outer signal layer.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Mask {
    pub thickness: f64,
    pub relative_permittivity: f64,
    /// +1 when the mask lies above the layer (a top outer layer), −1 below.
    pub outward: f64,
    /// The laminate surface the mask covers between traces.
    pub surface_y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Geometry {
    pub y_low: f64,
    pub y_high: f64,
    pub x_low: f64,
    pub x_high: f64,
    /// Zero-potential plane at the bottom / top boundary row.
    pub plane_low: bool,
    pub plane_high: bool,
    /// Dielectric bands (y0, y1, εr) inside the domain, outside the signal band.
    pub bands: Vec<(f64, f64, f64)>,
    /// The signal copper band and what fills it between traces.
    pub signal_band: (f64, f64),
    pub fill: f64,
    pub mask: Option<Mask>,
    pub traces: Vec<Trace>,
    /// Smallest geometric feature, for default mesh sizes.
    pub feature: f64,
    /// Dielectric height between the signal layer and its farthest bounding plane.
    pub height: f64,
}

fn invalid(message: impl Into<String>) -> CrossSectionError {
    CrossSectionError::Invalid(message.into())
}

fn positive(value: f64, what: &str) -> Result<f64, CrossSectionError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!("{what} must be positive and finite")))
    }
}

fn permittivity(value: f64, what: &str) -> Result<f64, CrossSectionError> {
    if value.is_finite() && value >= 1.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "{what}: relative permittivity must be finite and at least 1"
        )))
    }
}

impl Geometry {
    pub fn build(section: &CrossSection) -> Result<Self, CrossSectionError> {
        if section.layers.is_empty() {
            return Err(invalid("the stackup has no layers"));
        }
        if section.conductors.is_empty() {
            return Err(invalid("no conductors"));
        }
        let mut names = std::collections::BTreeSet::new();
        for layer in &section.layers {
            if !names.insert(layer.name.as_str()) {
                return Err(invalid(format!("layer `{}` is listed twice", layer.name)));
            }
            let signal = matches!(
                layer.kind,
                StackLayerKind::Copper {
                    role: CopperRole::Signal,
                    ..
                }
            );
            if !(layer.thickness_m.is_finite()
                && (layer.thickness_m > 0.0 || signal && layer.thickness_m == 0.0))
            {
                return Err(invalid(format!(
                    "layer `{}`: thickness must be positive and finite",
                    layer.name
                )));
            }
            match layer.kind {
                StackLayerKind::Dielectric {
                    relative_permittivity,
                }
                | StackLayerKind::SolderMask {
                    relative_permittivity,
                } => {
                    permittivity(relative_permittivity, &layer.name)?;
                }
                StackLayerKind::Copper {
                    fill_relative_permittivity: Some(fill),
                    ..
                } => {
                    permittivity(fill, &layer.name)?;
                }
                StackLayerKind::Copper { .. } => {}
            }
        }
        let layer_name = &section.conductors[0].layer;
        if section.conductors.iter().any(|c| &c.layer != layer_name) {
            return Err(invalid(
                "every conductor must be on the same signal layer in this version; broadside-coupled lines on two layers are a separate capability",
            ));
        }
        let index = section
            .layers
            .iter()
            .position(|layer| &layer.name == layer_name)
            .ok_or_else(|| {
                invalid(format!(
                    "conductor layer `{layer_name}` is not in the stackup"
                ))
            })?;
        let (fill_override, signal_thickness) = match section.layers[index].kind {
            StackLayerKind::Copper {
                role: CopperRole::Signal,
                fill_relative_permittivity,
            } => (
                fill_relative_permittivity,
                section.layers[index].thickness_m,
            ),
            _ => {
                return Err(invalid(format!(
                    "conductor layer `{layer_name}` must be a signal copper layer"
                )))
            }
        };
        // y up from the bottom of the stack.
        let total: f64 = section.layers.iter().map(|l| l.thickness_m).sum();
        let mut tops = Vec::with_capacity(section.layers.len());
        let mut y = total;
        for layer in &section.layers {
            tops.push(y);
            y -= layer.thickness_m;
        }
        let bottom_of = |i: usize| tops[i] - section.layers[i].thickness_m;
        let is_plane = |i: usize| {
            matches!(
                section.layers[i].kind,
                StackLayerKind::Copper {
                    role: CopperRole::ReferencePlane,
                    ..
                }
            )
        };
        let plane_above = (0..index).rev().find(|&i| is_plane(i));
        let plane_below = (index + 1..section.layers.len()).find(|&i| is_plane(i));
        let signal_band = (bottom_of(index), tops[index]);
        // Dielectric layers on each side up to the bounding plane or the board surface.
        let mask_layer = |ids: Vec<usize>| {
            ids.into_iter().find_map(|i| match section.layers[i].kind {
                StackLayerKind::SolderMask {
                    relative_permittivity,
                } => Some((section.layers[i].thickness_m, relative_permittivity)),
                _ => None,
            })
        };
        let above: Vec<usize> = (plane_above.map_or(0, |p| p + 1)..index).collect();
        let below: Vec<usize> = (index + 1..plane_below.unwrap_or(section.layers.len())).collect();
        let dielectric_count = |ids: &[usize]| {
            ids.iter()
                .filter(|&&i| matches!(section.layers[i].kind, StackLayerKind::Dielectric { .. }))
                .count()
        };
        let outer_top = plane_above.is_none() && dielectric_count(&above) == 0;
        let outer_bottom = plane_below.is_none() && dielectric_count(&below) == 0;
        if outer_top && outer_bottom {
            return Err(invalid("the signal layer has no dielectric on either side"));
        }
        let mut bands = Vec::new();
        for &i in above.iter().chain(&below) {
            match section.layers[i].kind {
                StackLayerKind::Dielectric {
                    relative_permittivity,
                } => bands.push((bottom_of(i), tops[i], relative_permittivity)),
                // A copper layer between the signal layer and its plane carries no conductor of
                // this section: its thickness is filled like the dielectric next to it.
                StackLayerKind::Copper {
                    fill_relative_permittivity,
                    ..
                } => {
                    let neighbour = (i > 0)
                        .then(|| &section.layers[i - 1].kind)
                        .and_then(|kind| match kind {
                            StackLayerKind::Dielectric {
                                relative_permittivity,
                            } => Some(*relative_permittivity),
                            _ => None,
                        })
                        .or_else(|| {
                            section.layers.get(i + 1).and_then(|l| match l.kind {
                                StackLayerKind::Dielectric {
                                    relative_permittivity,
                                } => Some(relative_permittivity),
                                _ => None,
                            })
                        })
                        .unwrap_or(1.0);
                    bands.push((
                        bottom_of(i),
                        tops[i],
                        fill_relative_permittivity.unwrap_or(neighbour),
                    ));
                }
                // Solder mask is applied conformally on the outer signal layer only; a mask
                // beyond other copper is part of the outer surface and modelled as a band.
                StackLayerKind::SolderMask {
                    relative_permittivity,
                } if !(outer_top || outer_bottom) => {
                    bands.push((bottom_of(i), tops[i], relative_permittivity))
                }
                StackLayerKind::SolderMask { .. } => {}
            }
        }
        let mask = if outer_top {
            mask_layer(above.iter().rev().copied().collect()).map(|(t, er)| Mask {
                thickness: t,
                relative_permittivity: er,
                outward: 1.0,
                surface_y: signal_band.0,
            })
        } else if outer_bottom {
            mask_layer(below.clone()).map(|(t, er)| Mask {
                thickness: t,
                relative_permittivity: er,
                outward: -1.0,
                surface_y: signal_band.1,
            })
        } else {
            None
        };
        let fill = match fill_override {
            Some(fill) => fill,
            None if outer_top || outer_bottom => 1.0,
            None => {
                let next = |ids: Vec<usize>| {
                    ids.into_iter().find_map(|i| match section.layers[i].kind {
                        StackLayerKind::Dielectric {
                            relative_permittivity,
                        } => Some(relative_permittivity),
                        _ => None,
                    })
                };
                next(above.iter().rev().copied().collect())
                    .or_else(|| next(below.clone()))
                    .unwrap_or(1.0)
            }
        };
        // Traces.
        let mut traces = Vec::with_capacity(section.conductors.len());
        let mut conductor_names = std::collections::BTreeSet::new();
        for conductor in &section.conductors {
            if !conductor_names.insert(conductor.name.as_str()) {
                return Err(invalid(format!(
                    "conductor `{}` is listed twice",
                    conductor.name
                )));
            }
            let width = positive(conductor.width_m, &format!("{} width", conductor.name))?;
            if !conductor.center_m.is_finite() {
                return Err(invalid(format!(
                    "{} position must be finite",
                    conductor.name
                )));
            }
            let angle = conductor.etch_angle_rad;
            if !(angle.is_finite() && angle > 0.0 && angle <= std::f64::consts::FRAC_PI_2 + 1e-12) {
                return Err(invalid(format!(
                    "{} etch angle must be in (0°, 90°]",
                    conductor.name
                )));
            }
            let inset = if angle >= std::f64::consts::FRAC_PI_2 {
                0.0
            } else {
                signal_thickness / angle.tan()
            };
            if width - 2.0 * inset <= 0.0 {
                return Err(invalid(format!(
                    "{}: the etch angle leaves no far face (width {width} m, thickness {signal_thickness} m)",
                    conductor.name
                )));
            }
            let base = conductor.base.unwrap_or(if outer_bottom {
                TraceBase::Top
            } else {
                TraceBase::Bottom
            });
            let (base_y, far_y) = match base {
                TraceBase::Bottom => signal_band,
                TraceBase::Top => (signal_band.1, signal_band.0),
            };
            let left = conductor.center_m - width / 2.0;
            let right = conductor.center_m + width / 2.0;
            traces.push(Trace {
                name: conductor.name.clone(),
                role: conductor.role,
                base_left: left,
                base_right: right,
                far_left: left + inset,
                far_right: right - inset,
                base_y,
                far_y,
            });
        }
        traces.sort_by(|a, b| a.base_left.total_cmp(&b.base_left));
        let mut feature = f64::INFINITY;
        if signal_thickness > 0.0 {
            feature = feature.min(signal_thickness);
        }
        for pair in traces.windows(2) {
            let gap =
                (pair[1].base_left - pair[0].base_right).min(pair[1].far_left - pair[0].far_right);
            if gap <= 0.0 {
                return Err(invalid(format!(
                    "conductors `{}` and `{}` overlap",
                    pair[0].name, pair[1].name
                )));
            }
            feature = feature.min(gap);
        }
        for trace in &traces {
            feature = feature.min(trace.far_right - trace.far_left);
        }
        if !traces.iter().any(|t| t.role == ConductorRole::Signal) {
            return Err(invalid("no signal conductor"));
        }
        let has_ground = traces.iter().any(|t| t.role == ConductorRole::Ground);
        if plane_above.is_none() && plane_below.is_none() && !has_ground {
            return Err(invalid(
                "the line has no return conductor: add a reference plane or coplanar ground",
            ));
        }
        let thinnest_band = bands
            .iter()
            .map(|(y0, y1, _)| y1 - y0)
            .fold(f64::INFINITY, f64::min);
        feature = feature.min(thinnest_band);
        if let Some(mask) = &mask {
            feature = feature.min(mask.thickness);
        }
        let height_above = plane_above.map(|p| bottom_of(p) - signal_band.1);
        let height_below = plane_below.map(|p| signal_band.0 - tops[p]);
        let side_height = |ids: &[usize]| {
            ids.iter()
                .map(|&i| section.layers[i].thickness_m)
                .sum::<f64>()
        };
        let height = match (height_above, height_below) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => side_height(&above).max(side_height(&below)),
        };
        let span = (
            traces
                .iter()
                .map(|t| t.base_left.min(t.far_left))
                .fold(f64::INFINITY, f64::min),
            traces
                .iter()
                .map(|t| t.base_right.max(t.far_right))
                .fold(f64::NEG_INFINITY, f64::max),
        );
        let enclosure = match section.enclosure {
            Some(enclosure) => {
                positive(enclosure.half_width_m, "enclosure half width")?;
                positive(enclosure.air_m, "enclosure air height")?;
                enclosure
            }
            None => Enclosure::default_for(height, span.1 - span.0),
        };
        let centre = 0.5 * (span.0 + span.1);
        let half = enclosure
            .half_width_m
            .max(0.5 * (span.1 - span.0) + 2.0 * height);
        let mask_reach = mask.as_ref().map_or(0.0, |m| m.thickness);
        let y_low = match plane_below {
            Some(p) => tops[p],
            None if outer_bottom => signal_band.0 - mask_reach - enclosure.air_m,
            None => bottom_of(*below.last().expect("dielectric below")) - enclosure.air_m,
        };
        let y_high = match plane_above {
            Some(p) => bottom_of(p),
            None if outer_top => signal_band.1 + mask_reach + enclosure.air_m,
            None => tops[*above.first().expect("dielectric above")] + enclosure.air_m,
        };
        Ok(Self {
            y_low,
            y_high,
            x_low: centre - half,
            x_high: centre + half,
            plane_low: plane_below.is_some(),
            plane_high: plane_above.is_some(),
            bands,
            signal_band,
            fill,
            mask,
            traces,
            feature,
            height,
        })
    }

    pub fn signal_names(&self) -> Vec<String> {
        self.traces
            .iter()
            .filter(|t| t.role == ConductorRole::Signal)
            .map(|t| t.name.clone())
            .collect()
    }

    /// Relative permittivity at a point that is not inside a conductor.
    pub fn relative_permittivity(&self, x: f64, y: f64) -> f64 {
        if let Some(mask) = &self.mask {
            if self.in_mask(mask, x, y) {
                return mask.relative_permittivity;
            }
        }
        if y >= self.signal_band.0 && y <= self.signal_band.1 {
            return self.fill;
        }
        self.bands
            .iter()
            .find(|(y0, y1, _)| y >= *y0 && y <= *y1)
            .map_or(1.0, |(_, _, er)| *er)
    }

    fn in_mask(&self, mask: &Mask, x: f64, y: f64) -> bool {
        // Depth outward from the laminate surface.
        let depth = (y - mask.surface_y) * mask.outward;
        if depth < 0.0 {
            return false;
        }
        if depth <= mask.thickness {
            return true;
        }
        self.traces.iter().any(|trace| {
            let height = (trace.far_y - trace.base_y).abs();
            if depth > height + mask.thickness {
                return false;
            }
            // Offset the trapezoid outward by the mask thickness: sidewalls move by t / sin(θ)
            // horizontally, the far face by t vertically.
            let (left, right) = trace.edges_at(mask.surface_y + mask.outward * depth.min(height));
            let slope_left = trace.far_left - trace.base_left;
            let sin = if height == 0.0 || slope_left == 0.0 {
                1.0
            } else {
                height / slope_left.hypot(height)
            };
            let shift = mask.thickness / sin;
            x >= left - shift && x <= right + shift
        })
    }

    /// Key x positions of the mesh: trace corners at the base and far faces, and the walls.
    pub fn x_keys(&self) -> Vec<KeyColumn> {
        let mut keys = Vec::new();
        let (band_lo, band_hi) = self.signal_band;
        let at = |base_y: f64, base: f64, far: f64| {
            if base_y == band_lo {
                (base, far)
            } else {
                (far, base)
            }
        };
        for trace in &self.traces {
            let (lo, hi) = at(trace.base_y, trace.base_left, trace.far_left);
            keys.push(KeyColumn::slanted(lo, hi, band_lo, band_hi, true));
            let (lo, hi) = at(trace.base_y, trace.base_right, trace.far_right);
            keys.push(KeyColumn::slanted(lo, hi, band_lo, band_hi, true));
        }
        keys.push(KeyColumn::slanted(
            self.x_low, self.x_low, band_lo, band_hi, false,
        ));
        keys.push(KeyColumn::slanted(
            self.x_high,
            self.x_high,
            band_lo,
            band_hi,
            false,
        ));
        keys.sort_by(|a, b| a.low.total_cmp(&b.low));
        keys
    }

    /// Key y positions: the domain bounds, every band interface, the signal band faces and
    /// the mask surfaces. `true` marks lines that touch a trace edge.
    pub fn y_keys(&self) -> Vec<(f64, bool)> {
        let mut keys = vec![
            (self.y_low, false),
            (self.y_high, false),
            (self.signal_band.0, true),
            (self.signal_band.1, true),
        ];
        for (y0, y1, _) in &self.bands {
            keys.push((*y0, false));
            keys.push((*y1, false));
        }
        if let Some(mask) = &self.mask {
            let height = self.signal_band.1 - self.signal_band.0;
            keys.push((mask.surface_y + mask.outward * mask.thickness, false));
            keys.push((
                mask.surface_y + mask.outward * (height + mask.thickness),
                false,
            ));
        }
        keys.retain(|(y, _)| *y >= self.y_low && *y <= self.y_high);
        keys
    }
}

/// A key mesh column: vertical outside the signal band, straight from `low` to `high` across
/// it (a trace sidewall).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct KeyColumn {
    pub low: f64,
    pub high: f64,
    pub band: (f64, f64),
    /// The column is a trace edge (fine elements on both sides).
    pub edge: bool,
}

impl KeyColumn {
    fn slanted(low: f64, high: f64, band_lo: f64, band_hi: f64, edge: bool) -> Self {
        Self {
            low,
            high,
            band: (band_lo, band_hi),
            edge,
        }
    }

    pub fn x_at(&self, y: f64) -> f64 {
        let (lo, hi) = self.band;
        if hi <= lo || y <= lo {
            self.low
        } else if y >= hi {
            self.high
        } else {
            self.low + (self.high - self.low) * (y - lo) / (hi - lo)
        }
    }
}

impl Enclosure {
    /// Side walls and lid `ENCLOSURE_HEIGHTS` dielectric heights from the traces.
    pub fn default_for(height: f64, span: f64) -> Self {
        Self {
            half_width_m: 0.5 * span + ENCLOSURE_HEIGHTS * height,
            air_m: ENCLOSURE_HEIGHTS * height,
        }
    }
}
