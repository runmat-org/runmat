# Nonuniform thermal cooling: spatial qualification window

This is constant-property continuum evidence, not restart or family qualification.
The actual solved heated cube supplies the cold-bound initial field; no analytic field
is substituted into the consumer. Independent Fourier diffusion supplies the physical
reference. Backward-Euler time-error isolation is a measurement, not a new integrator.

The original Tet4 cells 1/2/4 witness remains executable and failed. Its one-cell mesh
has no free interior-x temperature nodes, so its solved departure is identically zero.
Its measured temperature errors were 0.2719906/0.1015273/0.0277192: observed orders
1.42169 and 1.87291. Four-cell heat-content error 0.1013223 exceeds the original 0.08
limit. Neither the solver nor any accuracy/resource limit is changed to hide this.

An independent one-dimensional consistent-P1 calculation explains the coarse heat
error. For h=1/n, sine modes have eigenvalues
`6(1-cos(j*pi*h))/(h²*(2+cos(j*pi*h)))`. Project the actually solved steady nodal
field `q*x*(1-x)/2` onto these modes and integrate its decayed P1 interpolation.
Compare to continuum heat `8*q/pi^4 * sum_odd exp(-j²*pi²*t)/j^4`.
At t=0.1 this gives relative heat errors 1/0.38501345/0.10121678/0.02548858
for cells 1/2/4/8. This is not another RunMat solve and does not assume that the 3D
tetrahedral mesh is exactly its 1D discretization.

The separately named resolved Tet4 witness uses cells 2/4/8, chosen from that independent
calculation before executing its 3D result. It keeps the original temperature/flux
orders, endpoint limits, heat/power reduction, temporal-contamination, quadrature and
energy gates. Passing it cannot turn the original cells 1/2/4 witness green. Tet10
keeps its original 1/2/4 sequence and limits.

2026-10-01 diagnostic verification: the independent P1 test, both-order ramp heating
and temperature-offset invariance pass; strict FEA library/tests lint passes. The
original Tet4 rate failure reproduces. Tet10 fails consumer solve acceptance with
reported residual norm `3.848518805768323e-12` before completing its spatial sequence.
The resolved Tet4 witness is still running. No thermal gate, normal-CI or qualification
pass follows, and the failed qualification draft is not retained as a verified slice.

The source-free cooling chain covers temperature, recovered element-mean heat flux,
heat content, prescribed-boundary power, cold workload/field identity and unchanged
producer bytes. It does not prove arbitrary variable properties, radiation, nonlinear
closure, actual process replacement, generated-field equivalence or measured resources.
