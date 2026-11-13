use super::schema::{BuildConfiguration, CrateFeatureInventory};

const KNOWN_RUNTIME_FEATURES: &[&str] = &[
    "blas-lapack",
    "blas-only",
    "gui",
    "interaction-test-hooks",
    "occt-native",
    "occt-wasm-host",
    "plot-core",
    "plot-web",
    "test-classes",
    "wgpu",
];

pub(super) fn build_configuration() -> BuildConfiguration {
    let enabled_features = [
        (cfg!(feature = "blas-lapack"), "blas-lapack"),
        (cfg!(feature = "blas-only"), "blas-only"),
        (cfg!(feature = "gui"), "gui"),
        (
            cfg!(feature = "interaction-test-hooks"),
            "interaction-test-hooks",
        ),
        (cfg!(feature = "occt-native"), "occt-native"),
        (cfg!(feature = "occt-wasm-host"), "occt-wasm-host"),
        (cfg!(feature = "plot-core"), "plot-core"),
        (cfg!(feature = "plot-web"), "plot-web"),
        (cfg!(feature = "test-classes"), "test-classes"),
        (cfg!(feature = "wgpu"), "wgpu"),
    ]
    .into_iter()
    .filter_map(|(enabled, name)| enabled.then_some(name))
    .collect();

    BuildConfiguration {
        architecture: std::env::consts::ARCH,
        operating_system: std::env::consts::OS,
        family: std::env::consts::FAMILY,
        pointer_width: usize::BITS as u8,
        endianness: if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        },
        crate_feature_inventory: CrateFeatureInventory {
            crate_name: "runmat-runtime",
            schema_version: 1,
            known_features: KNOWN_RUNTIME_FEATURES.to_vec(),
            enabled_features,
        },
    }
}

pub(super) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}
