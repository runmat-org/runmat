# Local hosted native-pair qualification

`qualify-native-paired-host.py` does not build Rust or grant itself permission.
The operator supplies an actual target-built `runmat-aot-runtime` static archive,
the matching Cargo `native-static-libs` diagnostic log, a built host
`runmat-aot-pack` executable and an explicit configuration. Coordinate with the
single-build owner before building these products. Retain matching source,
features, profile, compiler flags and archive digests with the build evidence.

The script supports local macOS only. It inventories the selected real clang
and ld, their non-system loader dependencies, exact SDK library/framework
locators and transitive `.tbd` paths. Direct dependencies retain their original
library/framework names but use physical SDK-relative stub locators, such as
`usr/lib/libSystem.B.tbd` and framework `Versions/A` or `Versions/C` images.
The stub's primary install-name must agree with that locator, as required by
the production TAPI owner. Exact locators are retained in qualification evidence
and rechecked on issuance rather than reconstructed from SDK alias names.
Physical closure bytes retain their SDK tree paths;
no staged symlink escapes the tree. It does not copy the whole SDK. SDK parsing
is conservative; a staged-link failure is retained, not repaired by falling
back to an ambient SDK. System loader/framework dependencies remain an explicit
hosted OS dependency, not a hermetic or cross-platform qualification.

Example operator configuration (review all grants and set accelerator bits to
the **actual target-built archive**, not the example values):

```json
{
  "schema_version": 1,
  "expected_packer_digest": "sha256:<digest selected from the matching host packer build receipt>",
  "pack": {
    "cpu_baseline": "arm_v8_a",
    "numerical_behavior": {"kind": "platform_accurate"},
    "optimization": "speed",
    "available_capabilities": ["HostRuntime", "NativeCode"],
    "services": []
  },
  "authorized_runtime_capabilities": ["HostRuntime", "NativeCode"],
  "additional_components": [],
  "runtime_capabilities": {
    "static_program_calls": true,
    "runtime_builtins": true,
    "plot_core": true,
    "dynamic_source_loading": false,
    "closed_world_linking": true,
    "accelerators": {"wgpu": true, "cuda": true}
  },
  "storage": {
    "metadata_bytes": 16777216,
    "archive": {
      "max_entries": 200000,
      "max_files": 100000,
      "max_expanded_bytes": 4294967296,
      "max_file_bytes": 536870912,
      "max_path_bytes": 4096,
      "max_component_bytes": 255,
      "max_compression_ratio": 1000
    }
  }
}
```

For x86_64 use `x86_64_v1`. CPU baseline is a build-owner obligation: confirm
the actual Rust compiler target flags before admitting it. This smoke cannot
establish a reproducible/tolerance numerical policy and refuses those policies.
For the macOS ARM Runtime, use the architectural baseline plus the platform's
required crypto features: `-C target-cpu=generic -C target-feature=+aes,+sha2`.
Record `rustc --print cfg` with those exact flags in the build receipt. Plain
`generic` removes features required by the Darwin dependencies; the compiler's
default Apple CPU enables additional instructions and is not this baseline.
The full selected identity includes the Darwin platform: this Runtime is not
an ARM Linux baseline qualification. Generated user objects retain the separate
selected Cranelift ISA policy. Archive probe equality does not measure its ISA.
The selected Runtime archive must fit the existing 512 MiB component bound.
Do not weaken that bound if an unoptimized debug-info archive is too large;
select and record an appropriate existing build profile/configuration.

On macOS, `/usr/bin/strip -S -x -o NEW_ARCHIVE ORIGINAL_ARCHIVE` performs a
platform relocatable-link packaging step while retaining global interfaces.
Keep the original, record the transformation and actual linker identity
(`strip -v` reports its `ld -keep_private_externs -r -S -x` invocation), and
recheck global defined/undefined interfaces and canonical probe/profile data.
This is not metadata-only: literal pools, relocation records and code layout
can change. Accept the resulting artifact only through the real selected
links and executions, not a size check or claimed byte identity.

Never use strip-all, `strip=symbols`, or a local-symbol discard operation that
leaves dangling relocations. In particular, the tested Rust1.90 LLVM
`llvm-strip --strip-debug --discard-all` route removed required local targets
from this Mach-O archive despite preserving its code bytes and globals; that
derivative is rejected. Neither packaging route establishes CPU-feature,
performance or cross-platform qualification.

The witness reports its actual lowered native capabilities, effects and service
demands before pack selection. Those demands, not builtin names, determine the
needed implementations and permissions. `functiontests(localfunctions)` reads
the frozen source/function metadata; it does not itself require dynamic source
compilation. Keep the application and all three controls unchanged. Prefer the
ordinary optimized Runtime without `source-evaluation` when the lowered graph
does not require that service. Actual compiled accelerator profile bits also
do not grant device access: add Accelerator permission only when required.

If a program really reaches typed DynamicEval, its compiler-bearing Runtime
requires the real `source-evaluation` feature and independent permission for
`dynamic_source_loading: true`. A pack containing that factory must declare
the service even when a particular program does not reach it: the packer
checks declaration against the actual archive symbol. This declaration does
not by itself authorize use or install generic environment, filesystem,
network or process services:

```json
{"kind":{"kind":"source_evaluation"},
 "abi":"sha256:<canonical native_source_evaluation_service_abi digest>",
 "behavior":"hosted_ambient"}
```

Obtain that ABI from the actual matching built packer's
`--host-probe-facts` output, not a new ABI label. That operator-tool mode emits a
typed schema-version-2 JSON object containing its actual linked Runtime host
target, complete ABI binding, canonical probe-byte digest, Runtime/catalog
fingerprints and canonical service-kind/ABI pairs. Clock, local environment,
workspace, filesystem and random services, and hosted Network use their own
Rust-owned factory ABIs. These facts describe interfaces, not permission or
proof that a particular target archive contains them. With `--archive PATH`
it separately inspects bounded archive target/probe/accelerator bytes without
executing the archive. The packer checks the actual factory symbol when later
producing the selected pack.
Set `dynamic_source_loading` only with independent operator permission. No
Clock or local-service inputs are synthesized by this harness. Explicit
`clock_inputs` and `local_service_inputs` in the operator configuration pass
through to the independently issued admission and are validated by Rust against
the selected product. The actual pack producer checks each requested factory
against the target archive before accepting its declaration.

The operator configuration defaults to `"profile": "hosted_native"` and rejects
other profiles before creating qualification output. This macOS path uses hosted
SDK and loader dependencies; selecting a different label cannot establish
deterministic deployment. The deterministic gate requires a content-addressed
Linux/static pack with the actual native implementation archives, independently
admitted toolchain/sysroot trees and invocation inputs, followed by real
application/test execution and separate-process reopening. SDK `.tbd` import
stubs do not supply those implementations. This tool's arithmetic smoke
qualifies only the local build tools, not RunMat execution or numerical
reproducibility.

Run preparation without issuing admission:

```sh
python3 scripts/development/qualify-native-paired-host.py \
  --runtime-archive /absolute/librunmat_aot_runtime.a \
  --native-static-libs-log /absolute/runtime-build.log \
  --operator-config /absolute/operator.json \
  --packer /absolute/runmat-aot-pack \
  --out /absolute/new-host-qualification-directory
```

Inspect `qualification.json`, `qualification-transcript.jsonl`,
`host-platform.json`, `measured-native-probe.json`, both tree inventories and
the selected pack. Before staging, the script executes the selected packer's
host-probe mode and compares its full target/OS/Mach-O/endianness/pointer-width
and complete ABI/probe-byte identity with the separately inspected archive.
It checks the packer executable against the independently selected build digest
and retains that digest in the evidence. The operator must select a packer built
from the matching current source, not hash an arbitrary old ambient binary and
call that source provenance. Probe equality does not establish archive execution
conformance or service permission. The smoke
really compiles volatile arithmetic, links through the staged toolchain and
sysroot with no PATH and executes the resulting image. It qualifies those
local tools only; it does not execute the RunMat Runtime or application/test
pair. `qualification.json` binds real source/object/image/transcript hashes and
the exact Runtime/log/config bytes. Failure retains its output directory.

After review, repeat the same command and directory with
`--issue-local-host-admission`. This is an explicit operator acceptance of the
measured archive/ABI/target and successful local tool evidence. The script
rechecks retained measurements and only then writes `host-admission.json`.
The pack cannot issue that grant, and no unexecuted smoke gets a success digest.

Finally run the actual paired witness, using another existing empty directory:

```sh
/absolute/matching-build/examples/native_paired \
  /absolute/qualification/pack.json /absolute/qualification/components \
  /absolute/qualification/host-admission.json /absolute/empty-pair-output
```

An optional fifth argument supplies a closed witness-mode JSON object. Omit it for the pure witness. Use the identical mode file for fresh execution, installation issuance and separate-process reopening; its source literals are part of the compiled program identity. These modes define the checks to execute, not evidence that they have passed.

| Mode | Configuration and required observations |
| --- | --- |
| `pure` | `{"mode":"pure"}`; arithmetic and the three semantic controls, without selected local or Clock inputs. |
| `clock` | `{"mode":"clock","inputs":null}` for hosted Clock, or supply the exact independently admitted `NativeClockInputs` object. Checks calendar time, monotonic sleep/elapsed behavior and declared process-CPU availability. |
| `deterministic_local` | Supply `environment_key`, `environment_value`, `environment_write_key`, `environment_write_value`, `language_seed`, `read_path`, `read_utf8`, `write_path` and `write_utf8`. The environment, initial file bytes and seed must match the independently admitted local inputs. Checks environment mutation, current/temp directory, file read/write/readback and the initially selected RNG stream. Requires deterministic deployment. |
| `hosted_external` | Supply `console_line`, `http_port` and `tcp_port`; ports must be fixed, distinct and nonzero. Checks console publication and exact loopback HTTP/TCP exchanges. The driver owns bounded listener/socket deadlines and cleanup. Requires hosted deployment. |
| `hosted_callback` | Supply `artifact_bundle`, `expected_bundle_digest`, `foreign_policy`, `expected_foreign_policy_digest`, `interface_name`, `artifact_identity`, `library_digest`, `input`, `offset` and `expected`. The pinned canonical native-interface bundle must expose the fixture's exact `fixture_apply(int32, callback(int32)->int32)->int32` ABI. Checks real foreign callback success and an originating callback exception through application/test execution and reopening. Requires hosted deployment and independently admitted environment, callback and dynamic-dispatch services. Leave `local_service_inputs` absent/null: hosted environment ports are ambient services, not deterministic virtual inputs. |

Raw Clock JSON or `null` input is also accepted. The external mode does not claim to exercise foreign callback routing; that requires a configured foreign adapter calling through the selected callback service. Internal function-handle calls do not establish that evidence. Network exchanges also do not establish raw-wire accounting for the separately defined returned-artifact egress limit.

The callback mode preserves generic `calllib`'s conservative requirements. A pinned library and ABI do not prove native implementation purity, so this mode does not erase environment or Unknown effects, claim deterministic deployment, or authorize every available service. Its complete service-demand check uses the canonical effect-to-service policy plus supplemental runtime services. The policy must select native process isolation so the witness exercises the isolated callback route. The policy and native bundle are embedded in both products; changed bytes, in-process policy or missing independent grants must reject rather than silently selecting a different adapter.

The first successful run creates, without replacement,
`native-paired-product-v1.json` and `native-paired-qualification-v1.json` in
the output directory. The product record contains the existing portable pair
manifest and completed link evidence; it grants no reopen authority.

To qualify durable reopening in a separate process, obtain one independently
issued `ExternalNativeInstallationAdmission` JSON record for each role. Those
trusted operator records bind the exact role image, selected product, program
link inputs, link build, storage limit and deployment qualification. Then run:

The existing witness can issue these records from explicit operator intents.
After reviewing the first successful product and execution, supply one
`ExternalNativeInstallationIntent` JSON file per role: schema version2,
`expected_product`, `expected_image`, `deployment_qualification`, `layout`,
`storage` and `loader_images`. Expected digests and deployment permission must
come from that independent operator review. The tool does not derive them from
the persisted pair and turn them into permission. Operator JSON may be formatted;
issued transport records use the canonical Rust codec.

The hosted profile does not promise byte-identical independent links. Use `--await-installation-review` in place of `--issue-installation` when the completed image must be reviewed after linking. Supply new, absent intent paths and otherwise the same arguments. The witness links and executes both roles, writes the product and qualification records, then retains the original completed-link owners for up to ten minutes while awaiting the two operator intents. Independently review those actual images, receipts, loader references, storage grant and layout; publish each intent atomically without replacing an existing file. The witness does not generate its own grants, reconstruct completed-link authority from persisted metadata, relink the image after review, or relax expected image hashes. Missing or mismatched intents fail without successful admission. Reopen the resulting admitted pair in a separate process as below.

When the selected tools reproduce identical image bytes, run a second fresh selected link in a different empty output directory, using the same qualified build inputs and independently expected image digests:

```sh
/absolute/matching-build/examples/native_paired \
  --issue-installation \
  /absolute/qualification/pack.json /absolute/qualification/components \
  /absolute/qualification/host-admission.json /absolute/second-empty-pair-output \
  /absolute/application-intent.json /absolute/test-intent.json \
  /absolute/application-installation-admission.json \
  /absolute/test-installation-admission.json
```

This mode issues records only after the application and all three semantic
controls complete and the exact linked image matches its independent intent.
It does not relabel a previously persisted record as a verified link. Admission
outputs are create-new; a later write failure may leave the first completed
record, just as a later paired-link failure may leave its first image. Compare
the two clean builds with the observer below, then reopen the issued second
product in a separate process:

```sh
/absolute/matching-build/examples/native_paired \
  --reopen \
  /absolute/qualification/pack.json /absolute/qualification/components \
  /absolute/qualification/host-admission.json /absolute/empty-pair-output \
  /absolute/empty-pair-output/native-paired-product-v1.json \
  /absolute/application-installation-admission.json \
  /absolute/test-installation-admission.json \
  /absolute/new-reopen-qualification.json \
  /absolute/witness-mode.json
```

Omit the final witness argument under the same rules as the original run. The
reopen path rebuilds the portable pair from the verified pack and frozen
source/test plan, checks the rebuilt manifest exactly, matches each physical
image to its separate installation admission, and runs the same application
and three controls through the ordinary private image/process owner. The new
qualification receipt also uses create-new semantics.

The witness uses production selected admission, compiler, paired linker,
private-image process owners and framed receiver. A passing local tool smoke
does not imply a passing pair. Preserve application/test failures and renew
only their affected inputs. Managed ledger peaks are not RSS, application
memory limits or a full-call finite-memory certificate. Linux, Windows and
deterministic/deployed qualification remain separate requirements.

Record physical image sizes, digests and typed OS-loader observations after a
successful execution:

```sh
python3 scripts/development/record-native-paired-product.py \
  --pair-output /absolute/empty-pair-output \
  --execution-receipt /absolute/new-reopen-qualification.json \
  --receipt-validator /absolute/matching-build/examples/native_paired \
  --out /absolute/new-native-pair-observation.json
```

Pass `--compare /absolute/prior-observation.json` for a later independent
build. This external observation neither grants execution nor decides loader
dependency closure from names alone.

The matching Rust witness validates the canonical qualification receipt before
observation. Comparison requires the same full paired manifest and per-role
canonical link recipe, build and input identities. It reports image-byte and
loader-dependency equality separately; two matching image hashes from different
source, policy or toolchain inputs are not a reproducible-build qualification.

## Content-addressed Linux build inputs

The pack producer accepts canonical toolchain/sysroot metadata and raw blob directories emitted by its existing `--selected-tree` mode. To produce a pack with those retained build inputs, pass all six inputs together:

```sh
/absolute/matching-build/runmat-aot-pack --selected-pack \
  --archive /absolute/librunmat_aot_runtime.a \
  --declaration /absolute/pack.declaration.json \
  --linker-tree-metadata /absolute/linker.metadata.json \
  --linker-tree-blobs /absolute/linker.blobs \
  --sysroot-tree-metadata /absolute/sysroot.metadata.json \
  --sysroot-tree-blobs /absolute/sysroot.blobs \
  --build-tree-grant /absolute/independent-storage-grant.json \
  --native-archive-declaration /absolute/native-archives.json \
  --manifest-out /absolute/new-pack.json \
  --component-root /absolute/new-components
```

`native-archives.json` is a nonempty, component-ID-sorted array of objects with exactly `id` and `path` fields. Paths resolve relative to that declaration's directory; every archive contains an actual native implementation, not an SDK import stub. The declaration must have no ambient library/framework inputs. Both build trees and the native archives share the caller's aggregate entry/file/byte grant, in addition to per-component limits. Canonical metadata and its admitted extents are checked before referenced blob materialization. Component digests describe bytes; they do not grant permission to execute a tool or admit its ABI.

This producer path currently accepts Linux targets only. Its output inventory does not prove that the final executable is statically closed: qualification must inspect the actual completed image and its loader/dependency closure, execute both roles and controls, and reopen the independently admitted images. In particular, content-addressed build-tool bytes alone do not establish deterministic runtime behavior or eliminate an ambient loader dependency.
