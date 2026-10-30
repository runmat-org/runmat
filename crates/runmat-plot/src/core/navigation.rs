use serde::{Deserialize, Serialize};

/// Mouse mappings. Legacy retains the combined CAD/DCC bindings used before
/// navigation settings existed. CAD uses middle orbit/right pan; DCC uses
/// Alt+left orbit, Alt+middle pan and Alt+right dolly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NavigationPreset {
    #[default]
    Legacy,
    Cad,
    Dcc,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OrbitCenter {
    #[default]
    ViewCenter,
    Origin,
    UnderCursor,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct NavigationOptions {
    pub preset: NavigationPreset,
    pub orbit_center: OrbitCenter,
    pub zoom_speed: f32,
    pub reverse_zoom: bool,
    pub zoom_to_cursor: bool,
    pub trackpad_pan: bool,
}

impl Default for NavigationOptions {
    fn default() -> Self {
        Self {
            preset: NavigationPreset::Legacy,
            orbit_center: OrbitCenter::ViewCenter,
            zoom_speed: 1.0,
            reverse_zoom: false,
            zoom_to_cursor: true,
            trackpad_pan: false,
        }
    }
}

impl NavigationOptions {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.zoom_speed.is_finite() || !(0.01..=100.0).contains(&self.zoom_speed) {
            return Err("navigation zoomSpeed must be finite and between 0.01 and 100");
        }
        Ok(())
    }
}

/// Off means a single-sample attachment, never a zero-sample texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Antialiasing {
    Off,
    X2,
    #[default]
    X4,
    X8,
}

impl Antialiasing {
    pub const fn samples(self) -> u32 {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }
    pub fn from_samples(samples: u32) -> Result<Self, &'static str> {
        match samples {
            1 => Ok(Self::Off),
            2 => Ok(Self::X2),
            4 => Ok(Self::X4),
            8 => Ok(Self::X8),
            _ => Err("anti-aliasing samples must be 1 (off), 2, 4 or 8"),
        }
    }
    pub fn supported_by_formats(
        self,
        color: wgpu::TextureFormatFeatures,
        depth: wgpu::TextureFormatFeatures,
    ) -> bool {
        let count = self.samples();
        color
            .allowed_usages
            .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
            && depth
                .allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
            && color.flags.sample_count_supported(count)
            && depth.flags.sample_count_supported(count)
            && (count == 1
                || color
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn antialiasing_matrix_requires_matching_depth_color_and_resolve() {
        let format = |flags| wgpu::TextureFormatFeatures {
            allowed_usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
            flags,
        };
        let all = wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2
            | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4
            | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X8
            | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE;
        for mode in [
            Antialiasing::Off,
            Antialiasing::X2,
            Antialiasing::X4,
            Antialiasing::X8,
        ] {
            assert_eq!(Antialiasing::from_samples(mode.samples()).unwrap(), mode);
            assert!(mode.supported_by_formats(format(all), format(all)));
            if mode != Antialiasing::Off {
                assert!(!mode.supported_by_formats(
                    format(all),
                    format(wgpu::TextureFormatFeatureFlags::empty())
                ));
                assert!(!mode.supported_by_formats(
                    format(all & !wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE),
                    format(all)
                ));
            }
        }
        for invalid in [0, 3, 16] {
            assert!(Antialiasing::from_samples(invalid).is_err());
        }
    }
}
