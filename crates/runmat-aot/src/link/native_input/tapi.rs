//! TAPI text-stub admission for exact, qualified macOS SDK link images.
//!
//! This checks the selected link image's target and install identity. The
//! qualified linker remains the authority for TAPI exports and reexports;
//! parsing a stub is not a substitute for the completed link receipt.
use serde::Deserialize;

use runmat_execution::host::{
    NativeArchitecture, NativeObjectFormat, NativeOperatingSystem, NativeTargetIdentity,
};
use runmat_package::NormalizedRelativePath;

use crate::AotResult;

use super::{invalid, NativeHostedDependency};

#[derive(Deserialize)]
enum TaggedDocument {
    #[serde(rename = "tapi-tbd")]
    Tbd(TapiV4),
}

#[derive(Deserialize)]
struct TapiV4 {
    #[serde(rename = "tbd-version")]
    version: u8,
    targets: Vec<String>,
    #[serde(rename = "install-name")]
    install_name: String,
}

pub(super) fn validate(
    bytes: &[u8],
    target: &NativeTargetIdentity,
    dependency: &NativeHostedDependency,
    locator: &NormalizedRelativePath,
) -> AotResult<()> {
    if target.operating_system != NativeOperatingSystem::Macos
        || target.object_format != NativeObjectFormat::MachO
        || target.pointer_width != 64
    {
        return Err(invalid(
            "TAPI stub requires a selected 64-bit macOS Mach-O target",
        ));
    }
    // A text stub is a sysroot file, never a pack component or ambient host
    // lookup. The tree owner already verified the exact bytes and locator.
    let expected_name = install_name_for_locator(dependency, locator)?;
    let target_name = match target.architecture {
        NativeArchitecture::Aarch64 => "arm64-macos",
        NativeArchitecture::X86_64 => "x86_64-macos",
    };
    if bytes.len() > 16 * 1024 * 1024 || !bytes.starts_with(b"--- !tapi-tbd\n") {
        return Err(invalid(
            "hosted TAPI stub has an unsupported header or extent",
        ));
    }
    let input = std::str::from_utf8(bytes).map_err(invalid)?;
    let mut primary = None;
    let mut count = 0usize;
    for document in serde_yaml::Deserializer::from_str(input) {
        count += 1;
        if count > 4096 {
            return Err(invalid("TAPI stub exceeds its document bound"));
        }
        let TaggedDocument::Tbd(image) = TaggedDocument::deserialize(document).map_err(invalid)?;
        if image.version != 4
            || image.targets.is_empty()
            || !valid_install_name(&image.install_name)
        {
            return Err(invalid(
                "TAPI image has unsupported version, target, or install name",
            ));
        }
        if primary.is_none() {
            primary = Some(image);
        }
    }
    let primary = primary.ok_or_else(|| invalid("TAPI stub has no primary image"))?;
    if primary.install_name != expected_name
        || !primary.targets.iter().any(|name| name == target_name)
    {
        return Err(invalid(
            "TAPI primary image differs from the qualified SDK locator or selected platform",
        ));
    }
    Ok(())
}

fn install_name_for_locator(
    dependency: &NativeHostedDependency,
    locator: &NormalizedRelativePath,
) -> AotResult<String> {
    let path = locator.as_str();
    let stem = path
        .strip_suffix(".tbd")
        .ok_or_else(|| invalid("TAPI locator must name a .tbd regular sysroot member"))?;
    let name = match dependency {
        NativeHostedDependency::Library(_) if path.starts_with("usr/lib/") => {
            format!("/{stem}.dylib")
        }
        NativeHostedDependency::Framework(name)
            if path.starts_with("System/Library/Frameworks/")
                && path.contains(&format!("/{name}.framework/"))
                && path.ends_with(&format!("/{name}.tbd")) =>
        {
            format!("/{stem}")
        }
        _ => {
            return Err(invalid(
                "TAPI locator differs from qualified dependency kind",
            ))
        }
    };
    if !valid_install_name(&name) {
        return Err(invalid("TAPI locator has no canonical system install name"));
    }
    Ok(name)
}

fn valid_install_name(name: &str) -> bool {
    (name.starts_with("/usr/lib/") || name.starts_with("/System/Library/Frameworks/"))
        && !name.split('/').any(|part| part == "." || part == "..")
        && !name.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::host::NativeAbi;

    const STUB: &str = r"--- !tapi-tbd
tbd-version: 4
targets: [arm64-macos, x86_64-macos]
install-name: '/usr/lib/libSystem.B.dylib'
exports:
  - targets: [arm64-macos]
    symbols: [_printf]
--- !tapi-tbd
tbd-version: 4
targets: [arm64-macos]
install-name: '/usr/lib/system/libcache.dylib'
exports:
  - targets: [arm64-macos]
    symbols: [_cache_get]
...
";

    fn target(architecture: NativeArchitecture) -> NativeTargetIdentity {
        NativeTargetIdentity::new(
            architecture,
            NativeOperatingSystem::Macos,
            64,
            NativeAbi::new("fixture-only-abi").unwrap(),
            NativeObjectFormat::MachO,
        )
        .unwrap()
    }

    #[test]
    fn v4_multi_document_stub_matches_exact_sdk_image_and_macos_architecture() {
        let locator = NormalizedRelativePath::new("usr/lib/libSystem.B.tbd").unwrap();
        for architecture in [NativeArchitecture::Aarch64, NativeArchitecture::X86_64] {
            validate(
                STUB.as_bytes(),
                &target(architecture),
                &NativeHostedDependency::Library("System".into()),
                &locator,
            )
            .unwrap();
        }
        let other = NormalizedRelativePath::new("usr/lib/libobjc.A.tbd").unwrap();
        assert!(validate(
            STUB.as_bytes(),
            &target(NativeArchitecture::Aarch64),
            &NativeHostedDependency::Library("objc".into()),
            &other,
        )
        .is_err());

        let framework = STUB.replacen(
            "/usr/lib/libSystem.B.dylib",
            "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
            1,
        );
        validate(
            framework.as_bytes(),
            &target(NativeArchitecture::Aarch64),
            &NativeHostedDependency::Framework("Metal".into()),
            &NormalizedRelativePath::new(
                "System/Library/Frameworks/Metal.framework/Versions/A/Metal.tbd",
            )
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn unsupported_version_or_platform_is_not_a_link_image() {
        let locator = NormalizedRelativePath::new("usr/lib/libSystem.B.tbd").unwrap();
        for changed in [
            STUB.replacen("tbd-version: 4", "tbd-version: 3", 1),
            STUB.replace("arm64-macos", "arm64-maccatalyst")
                .replace("x86_64-macos", "x86_64-maccatalyst"),
        ] {
            assert!(validate(
                changed.as_bytes(),
                &target(NativeArchitecture::Aarch64),
                &NativeHostedDependency::Library("System".into()),
                &locator,
            )
            .is_err());
        }
    }
}
