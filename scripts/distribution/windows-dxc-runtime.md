# Windows shader compiler distribution

The default Windows WGPU provider uses D3D12 with dynamic DXC. Each deployed
GPU-enabled executable has `dxcompiler.dll` and `dxil.dll` beside it. The
provider resolves these paths from the executable's installation directory,
not its working directory or a developer SDK on PATH. Workers in a separate
directory need the same dependency pair in that directory.

`windows-dxc-runtime.json` pins the official publisher archive, target-specific
DLL hashes, and all three license files supplied in that archive. The preparer
verifies those source-controlled identities on initial preparation and cache
reuse. It refuses damaged or unexpected cache files rather than overwriting
them or selecting another shader compiler.

If the publisher's release-download route fails, the preparer can use Node.js
and GitHub CLI to fetch the same archive through the GitHub asset API. CI
supplies a read-only workflow token. Both routes verify the pinned SHA512
before extracting the archive and checking individual files.

On a configured Windows build host:

```powershell
$dxc = & ./scripts/distribution/prepare-windows-dxc-runtime.ps1 `
  -Target x86_64-pc-windows-msvc -Destination artifacts/dxc-runtime
node scripts/runtime/stage-builtin-example-product.mjs `
  --binary target/x86_64-pc-windows-msvc/release/runmat.exe `
  --destination artifacts/windows-product `
  --dependency-dir "$env:VCPKG_ROOT/installed/x64-windows/bin" `
  --dependency-dir $dxc.BinDir --dependency-notice-dir $dxc.NoticeDir
```

Run acceptance tests against the staged executable, including GPU execution
from a different working directory with developer dependency directories
removed from PATH. Test harness executables can be staged the same way.
Running an executable directly from Cargo's output directory also requires
the dependency pair beside that executable.

The release archive retains the publisher's unmodified license files in
`THIRD_PARTY_NOTICES.d`, separately from the shared Rust runtime notice payload.
Updating the compiler requires reviewing the publisher release, refreshing
the pinned archive and file hashes, and qualifying the staged Windows GPU
product. Preparation does not change system-wide DLLs, PATH, RunMat's license,
or the selected compute backend.
