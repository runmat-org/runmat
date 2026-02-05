## Plotting depth, clip planes, and grid helpers

RunMat’s 3D plotting is designed to stay robust across wide scale ranges (e.g. LiDAR point clouds,
radar volumes, CFD surfaces, and animated datasets).

### Depth mode

The renderer supports two depth mappings:

- **Standard depth**: near maps to 0, far maps to 1.0.
  - Depth compare: `LessEqual`
  - Depth clear: `1.0`
- **Reversed‑Z**: near maps to 1.0, far maps to 0.
  - Depth compare: `GreaterEqual`
  - Depth clear: `0.0`

Reversed‑Z helps preserve depth precision at far distances (a common game‑engine technique).
It is the **default** depth mode for 3D plots in RunMat.

### Clip plane policy

For interactive 3D viewports, RunMat can update near/far every frame based on the **visible scene
bounds**:

- Keeps `near` as large as possible (improves depth precision and reduces z‑fighting).
- Keeps `far` only as large as needed (avoids clipping surprises without sacrificing precision).

This is “CAD‑like”: the view should not unexpectedly clip away large portions of the scene when the
data scale changes or the camera moves.

This **dynamic** clip policy is enabled by default for 3D.

### 3D grid plane

The XY grid at \(z=0\) is rendered as a **procedural plane helper**:

- Depth‑tested so geometry can occlude it.
- No depth writes so it never occludes geometry.
- Uses derivatives in the shader for stable line density (avoids far‑plane popping from extremely
  large, CPU‑generated line meshes).

