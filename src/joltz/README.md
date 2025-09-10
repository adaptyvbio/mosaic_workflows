### Joltz
`joltz` is a straightforward translation of [boltz-1 (and boltz-2!)](https://github.com/jwohlwend/boltz) from pytorch to JAX, which is compatible with all the nice features of JAX (JIT/vmap/etc).

This is primarily used for protein design using hallucination see [boltz-binder-design](https://github.com/escalante-bio/boltz-binder-design).

For a bare-bones example of how to load and use the model see the [example script](example.py). In this repository, higher-level usage goes through `mosaic.losses.boltz.load_boltz` and `mosaic_workflows`.

```
from mosaic.losses.boltz import load_boltz, make_binder_features, Boltz1Loss
import mosaic.losses.structure_prediction as sp

joltz = load_boltz()
features, _ = make_binder_features(binder_len=20, target_sequence="MFEARLVQGSI", use_msa=False, use_msa_server=False)
loss = Boltz1Loss(
  joltz1=joltz,
  name="boltz1",
  loss=1.0 * sp.BinderTargetContact(contact_distance=21.0) + (-0.3) * sp.HelixLoss(),
  features=features,
  recycling_steps=0,
  deterministic=True,
)
```

End-to-end examples: `scripts/run_pdl1_boltzdesign_control.py`, `scripts/run_binder_games_boltz1_minmax.py`.

Work in progress, collaboration/feedback/PRs welcome!

Tested with boltz 2.0.3; will almost certainly break with more recent versions.

#### TODO:
- [ ] Chunking ?
- [ ] Replace dictionaries with `eqx.Module`s
- [ ] Tastefully sprinkle some `jax.lax.stop_grad`s in Boltz-2
- [ ] Finish boltz-2 confidence module
- [ ] Implement affinity module






