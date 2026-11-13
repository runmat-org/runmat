#!/usr/bin/env python3
"""Explicit local macOS tool qualification; never a target-execution certificate.

Consumes an operator configuration, a real target-built Runtime archive and its
native-static-libs log. Produces bounded physical trees, a real staged-tool
arithmetic compile/link/run transcript and a selected pack. Admission is only
published with the explicit operator issuance flag after every check succeeds.
No Cargo commands, downloads, SDK-wide copy or ambient selected-link override.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def hosted_profile(config):
    profile = config.get("profile", "hosted_native")
    if profile != "hosted_native":
        raise ValueError(
            "this macOS qualification uses hosted SDK/loader dependencies and "
            "only admits hosted_native; deterministic_deployed requires the "
            "separate content-addressed static Linux qualification"
        )
    return profile


def write_new(path, data):
    with path.open("xb") as output:
        output.write(data)


def validate_service_abis(services, facts):
    """Compare typed wire identities emitted by Rust, without guessing ABI names.

    This checks declarations only. Production pack creation independently
    verifies each requested factory in the actual target-built archive.
    """
    known = facts["service_abis"]
    for service in services:
        expected = {"kind": service["kind"], "abi": service["abi"]}
        if sum(item == expected for item in known) != 1:
            raise ValueError("service declaration differs from canonical Rust factory ABI facts")


def sdk_member(sdk, relative):
    """Return a physical SDK-relative locator, refusing SDK symlink escapes."""
    physical = (sdk / relative).resolve(strict=True)
    if not physical.is_file() or not physical.is_relative_to(sdk):
        raise ValueError(f"SDK member escapes its qualified root: {relative}")
    return physical.relative_to(sdk)


def validate_stub_locator(path, dependency):
    """Preflight the exact install-name relationship enforced by the Rust owner.

    This deliberately does not replace production TAPI target/schema admission.
    The completed selected link still validates the full canonical stub contract.
    """
    relative = Path(dependency["path"])
    if relative.suffix != ".tbd":
        return
    stem = str(relative.with_suffix(""))
    if dependency["kind"] == "library" and stem.startswith("usr/lib/"):
        expected = f"/{stem}.dylib"
    elif (dependency["kind"] == "framework"
          and stem.startswith("System/Library/Frameworks/")
          and f"/{dependency['name']}.framework/" in stem
          and relative.name == f"{dependency['name']}.tbd"):
        expected = f"/{stem}"
    else:
        raise ValueError(f"SDK locator differs from its dependency kind: {dependency}")
    text = path.read_text()
    if not text.startswith("--- !tapi-tbd\n"):
        raise ValueError(f"unsupported SDK stub header: {path}")
    primary = text.split("\n--- !tapi-tbd", 1)[0]
    install = re.search(r"(?m)^install-name:\s*['\"]?([^'\"\s]+)", primary)
    if install is None or install.group(1) != expected:
        raise ValueError(f"SDK primary install-name differs from its physical locator: {dependency}")


def direct_sdk_locator(sdk, item):
    name = item["name"]
    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", name):
        raise ValueError(f"unsafe native dependency name {name}")
    if item["kind"] == "library":
        relative = Path("usr/lib") / f"lib{name}.tbd"
        if not (sdk / relative).exists():
            relative = relative.with_suffix(".a")
    elif item["kind"] == "framework":
        relative = Path("System/Library/Frameworks") / f"{name}.framework" / f"{name}.tbd"
    else:
        raise ValueError(f"unsupported SDK dependency kind: {item}")
    relative = sdk_member(sdk, relative)
    result = {**item, "path": str(relative)}
    validate_stub_locator(sdk / relative, result)
    return result


class Qualification:
    def __init__(self, args):
        self.args = args
        self.config = json.loads(args.operator_config.read_text())
        if self.config["schema_version"] != 1:
            raise ValueError("unsupported operator configuration")
        self.profile = hosted_profile(self.config)
        self.root = args.out.resolve()
        self.reviewed_existing = self.root.exists() and args.issue_local_host_admission
        if not self.reviewed_existing:
            self.root.mkdir(exist_ok=False)
        self.transcript = self.root / "commands.jsonl"
        self.environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
        if self.config["pack"]["numerical_behavior"] != {"kind": "platform_accurate"}:
            raise ValueError("this host smoke does not qualify a reproducible/tolerance numerical policy")
        self.packer_digest = digest(args.packer.read_bytes())
        if self.config["expected_packer_digest"] != self.packer_digest:
            raise ValueError("packer differs from the independently selected current-source build digest")
        self.inventory = {}

    def run(self, command, *, env=None):
        process = subprocess.Popen(
            [str(item) for item in command], cwd=self.root,
            env=self.environment if env is None else env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=120)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            # The direct child is not reaped yet, so its process-group ID is
            # still reserved while retiring a nested compiler/linker too.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            with self.transcript.open("ab") as output:
                output.write(canonical({"command": [str(item) for item in command],
                    "failure": "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted",
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace")}) + b"\n")
            raise RuntimeError(f"command interrupted or timed out: {command}; see {self.transcript}") from error
        record = {"command": [str(item) for item in command],
                  "environment": self.environment if env is None else env,
                  "returncode": process.returncode,
                  "stdout": stdout.decode(errors="replace"),
                  "stderr": stderr.decode(errors="replace")}
        with self.transcript.open("ab") as output:
            output.write(canonical(record) + b"\n")
        if process.returncode:
            raise RuntimeError(f"command failed ({process.returncode}): {command}; see {self.transcript}")
        return stdout.decode(errors="strict")

    def physical(self, source, tree, relative):
        # Resolve input symlinks, but publish regular bytes under the original
        # logical SDK/tool locator. No staged symlink can escape its tree.
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"not a physical file: {source}")
        destination = tree / relative
        if destination.exists():
            if digest(destination.read_bytes()) != digest(source.read_bytes()):
                raise ValueError(f"conflicting staged locator {destination}")
            return False
        size = source.stat().st_size
        limits = self.config["storage"]["archive"]
        if size > min(limits["max_file_bytes"], 512 * 1024 * 1024):
            raise ValueError(f"file exceeds explicit storage/component bound: {source}")
        used = sum(item["bytes"] for item in self.inventory.values())
        if used + size > limits["max_expanded_bytes"]:
            raise ValueError("physical inventory exceeds operator byte grant")
        if len(self.inventory) + 1 > limits["max_files"]:
            raise ValueError("physical inventory exceeds operator file grant")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        executable = bool(source.stat().st_mode & 0o111)
        destination.chmod(0o700 if executable else 0o600)
        self.inventory[str(destination.relative_to(self.root))] = {
            "source": str(source), "bytes": size,
            "digest": digest(destination.read_bytes()), "executable": executable,
        }
        return True

    def tool_closure(self, clang, linker):
        tree = self.root / "toolchain"
        pending = [(clang, Path("bin/clang")), (linker, Path("bin/ld"))]
        observed_os_dependencies = set()
        while pending:
            source, relative = pending.pop()
            if not self.physical(source, tree, relative):
                continue
            dependencies = self.run(["/usr/bin/otool", "-L", source])
            for line in dependencies.splitlines():
                if not line.startswith("\t"):
                    continue
                dependency = line.strip().split(" (", 1)[0]
                if dependency.startswith("/usr/lib/") or dependency.startswith("/System/Library/"):
                    observed_os_dependencies.add(dependency)
                elif dependency.startswith("@rpath/"):
                    name = dependency.removeprefix("@rpath/")
                    physical = clang.parent.parent / "lib" / name
                    pending.append((physical, Path("lib") / name))
                elif dependency.startswith("@loader_path/"):
                    suffix = dependency.removeprefix("@loader_path/")
                    physical = (source.parent / suffix).resolve()
                    relative_dependency = Path(os.path.normpath(str(relative.parent / suffix)))
                    if ".." in relative_dependency.parts:
                        raise ValueError(f"loader dependency escapes tool tree: {dependency}")
                    pending.append((physical, relative_dependency))
                else:
                    raise ValueError(f"unsupported tool loader dependency: {dependency}")
        return tree, sorted(observed_os_dependencies)

    def native_inputs(self):
        lines = self.args.native_static_libs_log.read_text().splitlines()
        lines = [line.split("native-static-libs: ", 1)[1]
                 for line in lines if "native-static-libs: " in line]
        if not lines:
            raise ValueError("actual Cargo native-static-libs diagnostic is required")
        tokens = iter(shlex.split(lines[-1]))
        inputs = []
        for token in tokens:
            if token.startswith("-l") and len(token) > 2:
                inputs.append({"kind": "library", "name": token[2:]})
            elif token == "-framework":
                inputs.append({"kind": "framework", "name": next(tokens)})
            else:
                raise ValueError(f"native token has no selected typed locator: {token}")
        if not inputs:
            raise ValueError("empty native input diagnostic")
        return inputs

    def sdk_closure(self, sdk, inputs):
        tree = self.root / "sysroot"
        locators = []
        pending = []
        for item in inputs:
            # Keep the linker-requested name, but admit the physical stub's
            # canonical install-name locator (System -> System.B, iconv ->
            # iconv.2, Framework -> Versions/A/Framework), not its SDK alias.
            locator = direct_sdk_locator(sdk, item)
            pending.append(Path(locator["path"]))
            locators.append(locator)
        # ld may inspect SDK version metadata independently of .tbd parsing.
        for relative in [Path("SDKSettings.json"), Path("SDKSettings.plist"),
                         Path("System/Library/CoreServices/SystemVersion.plist")]:
            if (sdk / relative).is_file():
                sdk_member(sdk, relative)
                self.physical(sdk / relative, tree, relative)
        while pending:
            relative = pending.pop()
            sdk_member(sdk, relative)
            if not self.physical(sdk / relative, tree, relative):
                continue
            if relative.suffix != ".tbd":
                continue
            # Conservative cross-architecture union of actual install/reexport
            # paths. The later staged link proves sufficiency; never fall back
            # to ambient SDK files if an SDK format needs more inventory.
            text = (sdk / relative).read_text()
            for absolute in re.findall(r"/(?:usr/lib|System/Library/(?:Private)?Frameworks)/[A-Za-z0-9_+./-]+", text):
                dependency = Path(absolute.lstrip("/"))
                if dependency.suffix == ".dylib":
                    dependency = dependency.with_suffix(".tbd")
                elif dependency.suffix not in (".tbd", ".a"):
                    dependency = Path(str(dependency) + ".tbd")
                if (sdk / dependency).is_file():
                    pending.append(dependency)
        return tree, locators

    def produce_tree(self, name):
        tree = self.root / name
        entries = []
        for path in sorted(tree.rglob("*")):
            relative = str(path.relative_to(tree))
            if path.is_symlink():
                raise ValueError("staged tree unexpectedly contains a symlink")
            if path.is_dir():
                entries.append({"kind": "directory", "path": relative})
            else:
                entries.append({"kind": "file", "path": relative, "source": str(path),
                                "executable": bool(path.stat().st_mode & 0o111)})
        declaration = {"schema_version": 1, **self.config["storage"], "entries": entries}
        write_new(self.root / f"{name}.declaration.json", canonical(declaration))
        self.run([self.args.packer, "--selected-tree", "--declaration", self.root / f"{name}.declaration.json",
                  "--metadata-out", self.root / f"{name}.metadata.json", "--digest-out", self.root / f"{name}.digest.json",
                  "--blob-root", self.root / f"{name}.blobs"])
        metadata = json.loads((self.root / f"{name}.metadata.json").read_text())
        return {"metadata": f"{name}.metadata.json", "blobs": [
            {"component": item["component"], "path": f"{name}.blobs/{item['component']}"}
            for item in metadata["blobs"]]}

    def qualify(self):
        if self.reviewed_existing:
            self.issue()
            return
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "x86_64"):
            raise ValueError("this local qualification harness supports native macOS only")
        archive = self.args.runtime_archive.resolve(strict=True)
        if not archive.is_file() or archive.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("selected Runtime archive must be a physical file within 512 MiB")
        # Execute the actual host-linked probe separately from inspecting the
        # archive bytes. Refuse a same-architecture foreign OS/ABI before any
        # SDK staging, smoke, pack declaration or admission can be published.
        probe_facts = json.loads(self.run([self.args.packer, "--host-probe-facts", "--archive", archive]))
        if probe_facts["schema_version"] != 2 or probe_facts["archive"] is None:
            raise ValueError("missing actual host/archive probe measurement")
        host_target = probe_facts["host"]["target"]
        identity = host_target["target"]
        if (identity["architecture"] != ("aarch64" if platform.machine() == "arm64" else "x86_64")
                or identity["operating_system"] != "macos"
                or identity["object_format"] != "mach-o"
                or identity["pointer_width"] != 64
                or host_target["environment"] != "darwin"
                or host_target["endianness"] != "little"):
            raise ValueError("executed host probe differs from the actual local Darwin/Mach-O host")
        if probe_facts["archive"]["native"] != probe_facts["host"]:
            raise ValueError("Runtime archive target, complete ABI binding or probe bytes differ from the executed host probe")
        if probe_facts["archive"]["archive_digest"] != digest(archive.read_bytes()):
            raise ValueError("archive changed during actual probe inspection")
        if self.config["pack"]["cpu_baseline"] != host_target["cpu_baseline"]:
            raise ValueError("operator baseline differs from measured host probe")
        if self.config["runtime_capabilities"]["accelerators"] != probe_facts["archive"]["accelerators"]:
            raise ValueError("operator accelerator inventory differs from target-built profile bytes")
        validate_service_abis(self.config["pack"]["services"], probe_facts)
        write_new(self.root / "measured-native-probe.json", canonical(probe_facts))
        clang = Path(self.run(["/usr/bin/xcrun", "--find", "clang"]).strip()).resolve()
        linker = Path(self.run(["/usr/bin/xcrun", "--find", "ld"]).strip()).resolve()
        sdk = Path(self.run(["/usr/bin/xcrun", "--show-sdk-path"]).strip()).resolve()
        platform_facts = {
            "uname": self.run(["/usr/bin/uname", "-a"]),
            "os": self.run(["/usr/bin/sw_vers"]),
            "sdk_version": self.run(["/usr/bin/xcrun", "--show-sdk-version"]),
            "clang": self.run([clang, "--version"]),
            "ld": self.run([linker, "-v"]),
        }
        tools, os_dependencies = self.tool_closure(clang, linker)
        inputs = self.native_inputs()
        sysroot, locators = self.sdk_closure(sdk, inputs)
        source = self.root / "arithmetic.c"
        # Volatile operands prevent compile-time replacement of the oracle.
        write_new(source, b"int main(void) { volatile int x=6; volatile int y=7; return x*y == 42 ? 0 : 19; }\n")
        target = "aarch64-apple-darwin" if platform.machine() == "arm64" else "x86_64-apple-darwin"
        object_path = self.root / "arithmetic.o"
        image = self.root / "arithmetic"
        self.run([tools / "bin/clang", f"--target={target}", "-O0", "-c", source, "-o", object_path], env={"LC_ALL": "C"})
        self.run([tools / "bin/clang", f"--target={target}", f"--sysroot={sysroot}",
                  f"-fuse-ld={tools / 'bin/ld'}", "-nostdlib", "-nostartfiles", "-nodefaultlibs",
                  "-Wl,-t", object_path, *[sysroot / item["path"] for item in locators], "-o", image], env={"LC_ALL": "C"})
        self.run([image], env={"LC_ALL": "C"})
        self.produce_tree("toolchain")
        self.produce_tree("sysroot")
        tool_digest = json.loads((self.root / "toolchain.digest.json").read_text())
        sdk_digest = json.loads((self.root / "sysroot.digest.json").read_text())
        platform_facts["hosted_os_loader_dependencies"] = os_dependencies
        write_new(self.root / "host-platform.json", canonical(platform_facts))
        write_new(self.root / "qualification-transcript.jsonl", self.transcript.read_bytes())
        evidence = {"schema_version": 1, "scope": "local_hosted_clang_linker_arithmetic_smoke_only",
                    "platform": digest(canonical(platform_facts)), "toolchain": tool_digest, "sysroot": sdk_digest,
                    "runtime_archive": digest(archive.read_bytes()),
                    "native_static_libs_log": digest(self.args.native_static_libs_log.read_bytes()),
                    "source": digest(source.read_bytes()), "object": digest(object_path.read_bytes()),
                    "image": digest(image.read_bytes()),
                    "operator_config": digest(self.args.operator_config.read_bytes()),
                    "packer": self.packer_digest,
                    "native_probe": digest(canonical(probe_facts)),
                    "transcript": digest((self.root / "qualification-transcript.jsonl").read_bytes()),
                    "hosted_dependencies": locators,
                    "inventory": self.inventory}
        pack = {**self.config["pack"], "schema_version": 1,
                "platform_service_abi": evidence["platform"], "native_inputs": inputs,
                "linker_tree": tool_digest, "sysroot_tree": sdk_digest}
        write_new(self.root / "pack.declaration.json", canonical(pack))
        self.run([self.args.packer, "--selected-pack", "--archive", archive,
                  "--declaration", self.root / "pack.declaration.json",
                  "--manifest-out", self.root / "pack.json", "--component-root", self.root / "components"])
        measured = json.loads((self.root / "pack.json").read_text())
        if (measured["target"] != probe_facts["host"]["target"]
                or measured["abi"] != probe_facts["host"]["abi"]
                or measured["runtime"] != probe_facts["linked_runtime"]
                or measured["catalog"] != probe_facts["linked_catalog"]):
            raise ValueError("produced pack differs from independently executed host probe/runtime/catalog")
        evidence["pack_manifest"] = digest((self.root / "pack.json").read_bytes())
        evidence["pack_declaration"] = digest((self.root / "pack.declaration.json").read_bytes())
        write_new(self.root / "qualification.json", canonical(evidence))
        # The flag is the operator's explicit choice to accept these measured
        # runtime/ABI/target bytes and the independently successful host tools.
        # Without it there is no host-admission file and the pair cannot run.
        if self.args.issue_local_host_admission:
            self.issue()
        print(f"Local host smoke and selected pack complete: {self.root}")
        print("No RunMat application/test execution or deterministic-service qualification is claimed.")

    def issue(self):
        evidence_bytes = (self.root / "qualification.json").read_bytes()
        evidence = json.loads(evidence_bytes)
        checks = {
            self.args.runtime_archive: evidence["runtime_archive"],
            self.args.native_static_libs_log: evidence["native_static_libs_log"],
            self.args.operator_config: evidence["operator_config"],
            self.root / "qualification-transcript.jsonl": evidence["transcript"],
            self.root / "host-platform.json": evidence["platform"],
            self.root / "arithmetic.c": evidence["source"],
            self.root / "arithmetic.o": evidence["object"],
            self.root / "arithmetic": evidence["image"],
            self.root / "toolchain.metadata.json": evidence["toolchain"],
            self.root / "sysroot.metadata.json": evidence["sysroot"],
            self.root / "pack.json": evidence["pack_manifest"],
            self.root / "pack.declaration.json": evidence["pack_declaration"],
            self.args.packer: evidence["packer"],
            self.root / "measured-native-probe.json": evidence["native_probe"],
        }
        for path, expected in checks.items():
            if digest(path.read_bytes()) != expected:
                raise ValueError(f"reviewed qualification input changed: {path}")
        for relative, record in evidence["inventory"].items():
            path = self.root / relative
            if not path.resolve().is_relative_to(self.root) or path.is_symlink():
                raise ValueError(f"reviewed inventory path escapes: {relative}")
            if digest(path.read_bytes()) != record["digest"]:
                raise ValueError(f"reviewed staged input changed: {relative}")
        measured = json.loads((self.root / "pack.json").read_text())
        pack = json.loads((self.root / "pack.declaration.json").read_text())
        if measured["components"]:
            for component in measured["components"]:
                payload = self.root / "components" / component["id"]
                if digest(payload.read_bytes()) != component["object"]["digest"]:
                    raise ValueError("selected component changed after production")
        if measured["build_inputs"]["linker"]["fingerprint"] != evidence["toolchain"] or measured["build_inputs"]["sysroot"]["fingerprint"] != evidence["sysroot"]:
            raise ValueError("selected pack differs from independently measured tools")
        if digest((self.root / "components" / measured["runtime_component"]).read_bytes()) != evidence["runtime_archive"]:
            raise ValueError("selected Runtime differs from the measured archive")
        locators = evidence["hosted_dependencies"]
        if [{"kind": item["kind"], "name": item["name"]} for item in locators] != self.native_inputs():
            raise ValueError("reviewed SDK dependency names differ from actual native inputs")
        for locator in locators:
            relative = Path(locator["path"])
            staged = self.root / "sysroot" / relative
            if relative.is_absolute() or ".." in relative.parts or staged.is_symlink() or not staged.resolve().is_relative_to(self.root / "sysroot"):
                raise ValueError(f"reviewed SDK locator escapes its staged root: {locator}")
            validate_stub_locator(staged, locator)
        unique = {(item["kind"], item["name"]): item for item in locators}
        def tree_input(name):
            metadata = json.loads((self.root / f"{name}.metadata.json").read_text())
            return {"metadata": f"{name}.metadata.json", "blobs": [
                {"component": item["component"], "path": f"{name}.blobs/{item['component']}"}
                for item in metadata["blobs"]]}
        host = {"schema_version": 1, "expected_target": measured["target"], "expected_abi": measured["abi"],
                    "numerical": measured["numerical"], "profile": self.profile,
                    "platform_service_abi": evidence["platform"],
                    "service_abis": [{"kind": item["kind"], "abi": item["abi"]} for item in pack["services"]],
                    "authorized_runtime_capabilities": self.config["authorized_runtime_capabilities"],
                    "additional_components": self.config.get("additional_components", []),
                    "numerical_qualification": None, "runtime_capabilities": self.config["runtime_capabilities"],
                    "clock_inputs": self.config.get("clock_inputs"),
                    "local_service_inputs": self.config.get("local_service_inputs"),
                    "link": {"adapter": {"kind": "clang_driver", "linker": "bin/ld"}, "entrypoint": "bin/clang",
                             "toolchain": evidence["toolchain"], "sysroot": evidence["sysroot"],
                             "qualification": digest(evidence_bytes), "storage": self.config["storage"],
                             "hosted_dependencies": list(unique.values()),
                             "hosted_toolchain": tree_input("toolchain"), "hosted_sysroot": tree_input("sysroot")}}
        write_new(self.root / "host-admission.json", canonical(host))
        print(f"Operator-issued local hosted admission: {self.root / 'host-admission.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--native-static-libs-log", type=Path, required=True)
    parser.add_argument("--operator-config", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory, retained on failure")
    parser.add_argument("--issue-local-host-admission", action="store_true")
    args = parser.parse_args()
    args.packer = args.packer.resolve(strict=True)
    Qualification(args).qualify()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, subprocess.TimeoutExpired) as error:
        print(f"native host qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
