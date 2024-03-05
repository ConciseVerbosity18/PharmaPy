import numpy as np

from PharmaPy.Crystallizers import _BaseCryst
from PharmaPy.Reactors import _BaseReactor






class _BaseReactiveCryst(_BaseReactor,_BaseCryst):
    def __init__(self, mask_params, base_units, temp_ref,
     isothermal, reset_states, controls, h_conv, ht_mode,
      return_sens, state_events,method,target_comp,scale,
      vol_tank,adiabatic,rad_zero,vol_ht,basis,jac_type,
      param_wrapper):
        if isothermal:
            assert adiabatic != 1, "Cannot be isothermal and adiabatic with a reaction"
        _BaseReactor.__init__(self,mask_params, base_units, temp_ref,
         isothermal, reset_states, controls, h_conv, ht_mode,
          return_sens, state_events)
        _BaseCryst.__init__(self,mask_params,
                 method, target_comp, scale, vol_tank, controls,
                 adiabatic, rad_zero,
                 reset_states,
                 h_conv, vol_ht, basis, jac_type,
                 state_events, param_wrapper)

if __name__ == "__main__":
    RC = _BaseReactiveCryst(mask_params=None,
                 base_units='concentration', temp_ref=298.15,
                 isothermal=True, reset_states=False, controls=None,
                 h_conv=1000, ht_mode='jacket', return_sens=True,
                 state_events=None)
    print(RC.__class__.__mro__)