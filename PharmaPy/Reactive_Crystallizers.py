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
            assert adiabatic != 1, "Cannot be isothermal and adiabatic with a reaction present"
        _BaseCryst.__init__(self,mask_params,
                 method, target_comp, scale, vol_tank, controls,
                 adiabatic, rad_zero,
                 reset_states,
                 h_conv, vol_ht, basis, jac_type,
                 state_events, param_wrapper)
        # Reactor's init will overwrite any cryst commonalities, as keeping with the MRO
        # Beacause Kinetics is aliased in _BaseCryst, the reaction kinetics will become self.Kinetics and all
        # Crystallization kinetics must be accesed explicitly as CrystKinetics
        _BaseReactor.__init__(self,mask_params, base_units, temp_ref,
         isothermal, reset_states, controls, h_conv, ht_mode,
          return_sens, state_events)
        
        self.__original_prof__ = {
            'tempProf': [], 'concProf': [], 'timeProf': [],'distribProf': [],
            'volProf': [],
            'elapsed_time': 0, 'scale_flag':True
        }

    # TODO check that redefined kinetics account for both reaction kin and cryst kin
    # TODO use dimensionless?
    # TODO add material balances
    # TODO add energy balances
    # TODO create Reactive Cryst CSTR
        

