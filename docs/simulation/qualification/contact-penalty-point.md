# Penalty/Coulomb contact point operator — qualification boundary

This pure point law evaluates explicit normal and tangential penalties with the authored
friction model. It returns trial history and a local material tangent. It is not yet
coupled into FE equilibrium, surface search, global consistent tangent, restart, resource
admission, receipts or publication. Nonlinear preparation still rejects interfaces/contact.
Stage 6 and every original scientific gate remain open.

## Constitutive definition and original state

With positive-opening gap `g`, normal pressure is `p = kn * max(-g,0)`. Frictionless
controls have no tangential penalty; Coulomb controls require positive `kt` and finite
nonnegative coefficient `mu`. No numerical penalty default is introduced. An implicit
point update predicts `q_trial = q_accepted + kt * delta_s`, then returns radially to
the circular limit `|q| <= mu*p`. The zero-coefficient law has identically zero tangent
resistance, including at zero slip. Open contact releases its tangential spring and
retains accumulated original sliding/work; it does not count free-flight movement as
frictional slip. Reclosure begins with no residual elastic tangential traction.

Original state is opaque, serialized without defaults, and retains exact two-component
traction and net inelastic slip, accumulated sliding distance, frictional work, original
gap and law identity. Identity uses the shared canonical codec with explicit limits,
schema/domain and complete typed controls. Trial updates never mutate accepted state.
Cold validation checks finite history, its cone and net/accumulated-slip inequalities,
and exact law identity. It does not prove the past trajectory or authorize global restart;
mesh/interface/material-point/frame lineage belongs to the future complete state carrier.

Internal generalized traction is `[q1,q2,-p]`, conjugate to tangential material increments
and gap. In slip, the local tangential derivative is `kt*r/|q_trial| * (I-e*e^T)`;
its gap column is `-mu*kn*e`. The normal row has only `kn`. This tangent is generally
nonsymmetric. It holds accepted history and material frame fixed. FE assembly must
include pairing, weight, frame, normal and curvature derivatives; rotating this local
matrix alone is not the complete FE tangent. Open contact chooses the open derivative
at `g=0`; tests of smooth branches avoid the nonsmooth switching set.

## Objective material motion and energy

The point frame uses the original material tangent and current oriented normal through
the shared structural frame owner. Objective slip compares the SAME paired material
vectors in their accepted/current frames. A difference between freshly projected
closest-point gaps would erase actual tangential motion. Tracking paired material
locations across changing search features remains a full contact-operator requirement.
Frame normalization now rescales finite nonzero vectors before forming their norm;
there is no absolute length-unit cutoff, reciprocal overflow or squared-norm overflow.
This improves the shared owner used by loads and constraints as well as contact.

Normal/tangential elastic energy, backward-Euler Coulomb dissipation, tangential elastic
algorithmic loss, and spring energy discarded on separation are reported separately.
Algorithmic loss and separation release are not relabeled as frictional heat. Point
backward-Euler work is not an exact continuum path integral or global external-work balance.
Independent constant-pressure loading/reversal, pressure unloading, changing tangential
direction, energy identities, two-step-size full local Jacobians, proper rotations,
cold/tamper/foreign-law rejection and cancellation check this scope. Global normal/Hertz/
sliding/contact-parameter and mesh convergence evidence remains outstanding.

Method reference: [implicit penalty/friction trial and return mapping](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.088.html).
Repository verification scope and exact candidate receipts are recorded in the progress ledger.
