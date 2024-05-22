from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

from PharmaPy.Phases import classify_phases
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Commons import (reorder_sens, plot_sens, trapezoidal_rule,
                              upwind_fvm, high_resolution_fvm,
                              eval_state_events, handle_events,
                              unpack_states, complete_dict_states,
                              flatten_states)

from PharmaPy.ProcessControl import analyze_controls

from PharmaPy.jac_module import numerical_jac, numerical_jac_central, dx_jac_x
from PharmaPy.Connections import get_inputs, get_inputs_new

from PharmaPy.Results import DynamicResult
from PharmaPy.Plotting import plot_function, plot_distrib

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LightSource

from scipy.optimize import newton

import copy
import string
import numpy as np

eps = np.finfo(float).eps
# gas_ct = 8.314  # J/mol/K





class _BaseReactiveCryst():
    def __init__(self, mask_params, temp_ref,
     isothermal, reset_states, controls, h_conv, ht_mode,
      return_sens, state_events,method,target_comp,scale,
      vol_tank,adiabatic,rad_zero,vol_ht,basis,jac_type,
      param_wrapper):
        """ Construct a Reactive Crystallizer Object

    Parameters
    ----------
    mask_params : list of bool (optional, default = None)
        Binary list of which parameters to exclude from the kinetics
        computations
    method : str
        Choice of the numerical method. Options are: 'moments', '1D-FVM'
    target_comp : str, list of strings
        Name of the crystallizing compound(s) from .json file.
    scale : float
        Scaling factor by which crystal size distribution will be
        multiplied.
    vol_tank : TODO - Remove, it comes from Phases module.
    controls : dict of dicts(funcs) (optional, default = None)
        Dictionary with keys representing the state(e.g.'Temp') which is
        controlled and the value indicating the function to use
        while computing the variable. Functions are of the form
        f(time) = state_value
    adiabatic : bool (optional, default=True)
        Boolean value indicating whether the heat transfer of
        the crystallization is considered.
    rad_zero : float (optional, default=TODO)
        TODO Size of the first bin of the CSD discretization [m]
    reset_states : bool (optional, default = False)
        Boolean value indicating whether the states should be reset
        before simulation
    h_conv : float (optional, default = TODO) (maybe remove?)
        TODO
    vol_ht : float (optional, default = TODO)
        TODO Volume of the cooling jacket [m^3]
    basis : str (optional, default = T0DO)
        TODO Options :'massfrac', 'massconc'
    jac_type : str
        TODO Options: 'AD'
    state_events : lsit of dict(s)
        list of dictionaries, each one containing the specification of a
        state event
    param_wrapper : callable (optional, default = None)
        function with the signature

            param_wrapper(states, sens)

        Useful when the parameter estimation problem is a function of the
        states y -h(y)- rather than y itself.

        'states' is a DynamicResult object and 'sens' is a dictionary
        that contains N_y number of sensitivity arrays, representing
        time-depending sensitivities. Each array in sens has dimensions
        num_times x num_params. 'param_wrapper' has to return two outputs,
        one array containing h(y) and list of arrays containing
        sens(h(y))
    """

        if isothermal:
            assert adiabatic != 1, "Cannot be isothermal and adiabatic with a reaction present"
        self.mask_params = mask_params
        


        # Cryst first then reactor inits, but inherit reactor then cryst,
        #Reactor's init will overwrite any cryst commonalities, as keeping with the MRO
        # Beacause Kinetics is aliased in _BaseCryst, the reaction kinetics will become self.Kinetics and all
        # Crystallization kinetics must be accesed explicitly as CrystKinetics
        #from cryst
        self.basis = basis
        self.adiabatic = adiabatic
        self.jac_type = jac_type

        if isinstance(target_comp, str):
            target_comp = [target_comp]

        self.target_comp = target_comp

        self.scale = scale
        self.scale_flag = True
        self.vol_tank = vol_tank

        # Controls
        if controls is None:
            self.controls = {}
        else:
            self.controls = analyze_controls(controls) #was just controls for reactor Z

        self.method = method
        self.rad = rad_zero

        self.dx = None
        self.sensit = None

        self.jac_states_vals = None


        # Outlets
        self.reset_states = reset_states
        self.elapsed_time = 0

        self.profiles_runs = []

        self.__original_prof__ = {
            'tempProf': [], 'concProf': [], 'distribProf': [], 'timeProf': [],
            'elapsed_time': 0, 'scale_flag': True
        }

        # ---------- Names
        # TODO need states_uo, switch to mass on rxn side and convert
        self.states_uo_cryst = ['mass_conc'] #duplicate?
        self.names_states_in = ['mass_conc']

       

        self.names_upstream = None
        self.bipartite = None

        # Other parameters
        self.h_conv = h_conv

        # Slurry phase
        self.Slurry = None

        # Parameters for optimization
        self.params_iter = None
        self.vol_mult = 1

        if state_events is None:
            state_events = []

        self.state_event_list = state_events

        self.param_wrapper = param_wrapper

        self.outputs = None
        # Building objects
        self._CrystKinetics = None
        self._Utility = None
        self.material_from_upstream = False
        #------------from reactor
        self.distributed_uo = False
        self.is_continuous = False

        self.h_conv = h_conv
        self.area_ht = None
        self._Utility = None

        # Names
        self.bipartite = None
        self.names_upstream = None
        self.states_uo_reactor = ['mole_conc']
        self.names_states_out = ['mole_conc']
        self.states_out_dict = {}

        self.permut = None
        self.names_upstream = None

        self.ht_mode = ht_mode

        self.temp_ref = temp_ref

        self.isothermal = isothermal

        # ---------- Modeling objects
        self._Phases = None
        self._RxnKinetics = None
        self.mask_params = mask_params

        self.return_sens = return_sens

        # Outputs
        self.reset_states = reset_states

        self.__original_prof__ = {
            'tempProf': [], 'concProf': [], 'timeProf': [],'distribProf': [],
            'volProf': [],
            'elapsed_time': 0, 'scale_flag':True
        }
        
        self.temp_control = None
        self.resid_time = None
        self.oper_mode = None
        

        if state_events is None:
            state_events = []

        self.state_event_list = state_events

        # Outputs
        self.time_runs = []
        self.temp_runs = []
        self.conc_runs = []
        self.vol_runs = []
        self.tempHt_runs = []

        

        self.outputs = None
    # TODO check that redefined kinetics account for both reaction kin and cryst kin
    # TODO use dimensionless?
    # TODO add material balances
    # TODO add energy balances
    # TODO check if names_states or states_uo should be duplicate or singular
        

    @property
    def Phases(self):
        return self._Phases
    
    @Phases.setter
    def Phases(self, phases):
        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        elif isinstance(phases, Slurry):
            self._Phases = phases.Phases
        elif phases.__module__ == 'PharmaPy.Phases':
            if self._Phases is None:
                self._Phases = [phases]
            else:
                self._Phases.append(phases)
        else:
            raise RuntimeError('Please provide a list or tuple of phases '
                               'objects')
        
        if isinstance(phases, Slurry):
            self.Slurry = phases
        elif isinstance(self._Phases, (list, tuple)):
            if len(self._Phases) > 1:

                # Mixed phase
                self.Slurry = Slurry()
                self.Slurry.Phases = self._Phases

        if self.Slurry is not None:
            self.__original_phase_dict__ = [
                copy.deepcopy(phase.__dict__) for phase in self.Slurry.Phases]

            self.vol_slurry = copy.copy(self.Slurry.vol)
            if isinstance(self.vol_slurry, np.ndarray):
                self.vol_phase = self.vol_slurry[0]
            else:
                self.vol_phase = self.vol_slurry

            classify_phases(self)  # Solid_1, Liquid_1... Solid/Liquid/Vapor_n

            # Names and target compounds
            self.name_species = self.Liquid_1.name_species

            # Input defaults
            self.input_defaults = {
                'distrib': np.zeros_like(self.Solid_1.distrib)}

            name_bool = [name in self.target_comp for name in self.name_species]
            self.target_ind = np.where(name_bool)[0][0]

            # Save safe copy of original phases
            # TODO needs deep copy all phases for reset()
            self.__original_phase__ = [copy.deepcopy(self.Liquid_1),
                                       copy.deepcopy(self.Solid_1)]

            self.__original_phase__ = copy.deepcopy(self.Slurry)

            self.kron_jtg = np.zeros_like(self.Liquid_1.mass_frac) # this assumes Liquid_1 is the mother liquor
            self.kron_jtg[self.target_ind] = 1

            # ---------- Names
            # Moments
            # assumes solid 1 is target solid
            if self.method == 'moments':
                name_mom = [r'\mu_{}'.format(ind) for ind
                            in range(self.Solid_1.num_mom)]
                name_mom.append('C')

                self.num_distr = len(self.Solid_1.moments)

            else:
                self.num_distr = len(self.Solid_1.distrib)

            # Species
            if self.name_species is None:
                num_sp = len(self.Liquid_1.mass_frac)
                self.name_species = list(string.ascii_uppercase[:num_sp])

            self.states_in_dict = {
                'Liquid_1': {'mass_conc': len(self.Liquid_1.name_species)},
                'Inlet': {'vol_flow': 1, 'temp': 1}}

            self.nomenclature() 
            # TODO write nomenclature or set_names
            # TODO check if fine or need add support for other liquids present
            # TODO else fine
    
    @property
    def CrystKinetics(self):
        return self._CrystKinetics

    @CrystKinetics.setter
    def CrystKinetics(self, instance):
        self._CrystKinetics = instance

        name_params = self._CrystKinetics.name_params
        if self.mask_params is None:
            self.mask_params = [True] * self._CrystKinetics.num_params
            self.name_params = name_params

        else:
            self.name_params = [name for ind, name in enumerate(name_params)
                                if self.mask_params[ind]]

        self.mask_params = np.array(self.mask_params)

    @property
    def RxnKinetics(self):
        return self._RxnKinetics

    @RxnKinetics.setter
    def RxnKinetics(self, instance):
        self._RxnKinetics = instance
        self.partic_species = instance.partic_species

        name_params = self._RxnKinetics.name_params
        if self.mask_params is None:
            self.mask_params = [True] * self._RxnKinetics.num_params
            self.name_params = name_params

        else:
            self.name_params = [name for ind, name in enumerate(name_params)
                                if self.mask_params[ind]]

        self.mask_params = np.array(self.mask_params)

        ind_true = np.where(self.mask_params)[0]
        ind_false = np.where(~self.mask_params)[0]

        self.params_fixed = self.RxnKinetics.concat_params()[ind_false]

        self.ind_maskpar = np.argsort(np.concatenate((ind_true, ind_false)))
    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility

    def reset(self):
        copy_dict = copy.deepcopy(self.__original_prof__)
        self.__dict__.update(copy_dict)

        for phase, di in zip(self.Phases, self.__original_phase_dict__):
            phase.__dict__.update(di)

        self.profiles_runs = []
        # TODO does not work if more than 1 liquid and 1 solid
    
    def _eval_state_events(self, time, states, sw):
        # TODO reactor version changes discretized_model to True if PFR (cobc in our case)
        events = eval_state_events(
            time, states, sw, self.len_states,
            self.states_uo, self.state_event_list, sdot=self.derivatives,
            discretized_model=False)

        return events
    
    def heat_transfer(self, temp, temp_ht, vol):
        # Heat transfer area ##r
        heat_transf = self.u_ht * self.area_ht * (temp - temp_ht)
        return heat_transf
    
    def nomenclature(self):
        name_class = self.__class__.__name__

        states_di = {
            }

        di_distr = {'dim': self.num_distr,
                    'index': list(range(self.num_distr)), 'type': 'diff',
                    'depends_on': ['time', 'x_cryst']}

        if name_class != 'BatchCryst':
            self.names_states_in += ['vol_flow', 'temp']

            if self.method == 'moments':
                self.states_in_dict['Inlet']['mu_n'] = self.num_distr
            else:
                self.states_in_dict['Inlet']['distrib'] = self.num_distr

        if self.method == 'moments':
            # mom_names = ['mu_%s0' % ind for ind in range(self.num_mom)]

            # for mom in mom_names[::-1]:
            self.names_states_in.insert(0, 'mu_n')

            # self.states_in_dict['solid']['moments']

            if name_class == 'MSMPR':
                self.states_uo.append('moments')
                # self.states_in_dict['Inlet']['distrib'] = self.num_distr

                di_distr['units'] = 'm**n/m**3'
                states_di['mu_n'] = di_distr
            else:
                self.states_uo.append('total_moments')

                di_distr['units'] = 'm**n'
                states_di['mu_n'] = di_distr

                # if name_class == 'SemibatchCryst':
                    # self.states_in_dict['Inlet']['distrib'] = self.num_distr

        elif self.method == '1D-FVM':
            self.names_states_in.insert(0, 'distrib')

            states_di['distrib'] = di_distr

            if name_class == 'MSMPR':
                self.states_uo.insert(0, 'distrib')
                di_distr['units'] = '#/m**3/um'

            else:
                self.states_uo.insert(0, 'total_distrib')
                di_distr['units'] = '#/um'

        states_di['mass_conc'] = {'dim': len(self.name_species),
                                  'index': self.name_species,
                                  'units': 'kg/m**3', 'type': 'diff',
                                  'depends_on': ['time']}

        if name_class != 'MSMPR':
            states_di['vol'] = {'dim': 1, 'units': 'm**3', 'type': 'diff',
                                'depends_on': ['time']}
            self.states_uo.append('vol')

        if self.adiabatic:
            self.states_uo.append('temp')

            states_di['temp'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                 'depends_on': ['time']}
        elif 'temp' not in self.controls:
            self.states_uo += ['temp', 'temp_ht']

            states_di['temp'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                 'depends_on': ['time']}
            states_di['temp_ht'] = {'dim': 1, 'units': 'K', 'type': 'diff',
                                    'depends_on': ['time']}

        self.states_in_phaseid = {'mass_conc': 'Liquid_1'}
        self.names_states_out = self.names_states_in

        self.states_di = states_di
        self.dim_states = [di['dim'] for di in self.states_di.values()]
        self.name_states = list(self.states_di.keys())

        self.fstates_di = {
            'supersat': {'dim': 1, 'units': 'kg/m**3'},
            'solubility': {'dim': 1, 'units': 'kg/m**3'}
            }

        if 'temp' in self.controls:
            self.fstates_di['temp'] = {'dim': 1, 'units': 'K'}

        if self.method != 'moments':
            self.fstates_di['mu_n'] = {'dim': 4, 'index': list(range(4)),
                                       'units': 'm**n'}

            self.fstates_di['vol_distrib'] = {
                'dim': self.num_distr,
                'index': list(range(self.num_distr)),
                'units': 'm**3/m**3'}

    

    def get_inputs(self, time):
        ##r
        inlet = getattr(self, 'Inlet', None)
        if inlet is None:
            inputs = {}
        else:
            inputs = get_inputs_new(time, inlet, self.states_in_dict)

        return inputs
    
    def unit_model(self, time, states, params=None, sw=None,
                    mat_bce=False, enrgy_bce=False):
        # TODO reconcile with RC
        di_states = unpack_states(states, self.dim_states, self.name_states)

        # Inputs
        u_input = self.get_inputs(time)

        di_states = complete_dict_states(time, di_states,
                                        ('temp', 'temp_ht', 'vol'),
                                        self.Slurry, self.controls)

        # ---------- Physical properties
        self.Liquid_1.updatePhase(mass_conc=di_states['mass_conc'])
        self.Liquid_1.temp = di_states['temp']
        self.Solid_1.temp = di_states['temp']

        rhos_susp = self.Slurry.getDensity(temp=di_states['temp'])

        name_unit = self.__class__.__name__

        if self.method == 'moments':
            di_states['distrib'] = di_states['mu_n']
            moms = di_states['mu_n'] * \
                (1e-6)**np.arange(self.states_di['mu_n']['dim']) ###units

        else:
            moms = self.Solid_1.getMoments(
                distrib=di_states['distrib']/self.scale)  # m**n

        di_states['mu_n'] = moms

        if name_unit == 'BatchRC':
            rhos = rhos_susp
            h_in = None
            phis_in = None
        elif name_unit == 'SemibatchRC' or name_unit == 'RMSMPR':
            inlet_temp = u_input['Inlet']['temp']

            if self.Inlet.__module__ == 'PharmaPy.MixedPhases':
                rhos_in = self.Inlet.getDensity(temp=di_states['temp'])

                if 'distrib' in u_input['Inlet']:

                    inlet_distr = u_input['Inlet']['distrib']

                    mom_in = self.Inlet.Solid_1.getMoments(distrib=inlet_distr,
                                                            mom_num=3)
                elif 'mu_n' in u_input['Inlet']:

                    mom_in = np.array([u_input['Inlet']['mu_n'][3]])


                phi_in = 1 - self.Inlet.Solid_1.kv * mom_in
                phis_in = np.concatenate([phi_in, 1 - phi_in]) # TODO assumes only two pahses

                h_in = self.Inlet.getEnthalpy(inlet_temp, phis_in, rhos_in)
            else:
                rho_liq_in = self.Inlet.getDensity(temp=inlet_temp)
                rho_sol_in = None

                rhos_in = np.array([rho_liq_in, rho_sol_in])
                h_in = self.Inlet.getEnthalpy(temp=inlet_temp)

                phis_in = [1, 0]

            rhos = [rhos_susp, rhos_in]

        # Balances
        material_bces, cryst_rate = self.material_balances(
            time, params, u_input, rhos, **di_states, phi_in=phis_in)

        if mat_bce:
            return material_bces
        elif enrgy_bce:
            energy_bce = self.energy_balances(
                time, params, cryst_rate, u_input, rhos, **di_states,
                h_in=h_in, heat_prof=True)

            return energy_bce
        #equal here with ~R386
        else:

            if 'temp' in self.name_states:
                energy_bce = self.energy_balances(
                    time, params, cryst_rate, u_input, rhos, **di_states,
                    h_in=h_in)

                balances = np.append(material_bces, energy_bce)
            else:
                balances = material_bces

            self.derivatives = balances

            return balances
        # TODO jsut fix multiple states if needed else good
    
    def method_of_moments(self, mu, conc, temp, params, rho_cry, vol=1):
        kv = self.Solid_1.kv # shape factor

        # Kinetics
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        # Kinetic terms
        mu_susp = mu*(1e-6)**np.arange(self.num_distr) / vol  # m**n/m**3_susp
        nucl, growth, dissol = self.CrystKinetics.get_kinetics(comp_kin, temp, kv,
                                                          mu_susp)

        growth = growth * self.CrystKinetics.alpha_fn(conc)

        ind_mom = np.arange(1, len(mu))

        # Model
        dmu_zero_dt = np.atleast_1d(nucl * vol)
        dmu_1on_dt = ind_mom * (growth + dissol) * mu[:-1] + \
            nucl * self.rad**ind_mom
        dmu_dt = np.concatenate((dmu_zero_dt, dmu_1on_dt))

        # Material balance in kg_API/s --> G in um, u_2 in um**2 (or m**2/m**3)
        mass_transf = np.atleast_1d(rho_cry * kv * (
            3*(growth + dissol)*mu[2] + nucl*self.rad**3)) * (1e-6)**3

        return dmu_dt, mass_transf

    def fvm_method(self, csd, moms, conc, temp, params, rho_cry,
                   output='dstates', vol=1):

        mu_2 = moms[2]
        #assumes solid1 is target
        kv_cry = self.Solid_1.kv

        # Kinetic terms
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        nucl, growth, dissol = self.CrystKinetics.get_kinetics(comp_kin, temp,
                                                          kv_cry, moms)

        nucl = nucl * self.scale * vol

        impurity_factor = self.CrystKinetics.alpha_fn(conc)
        growth = growth * impurity_factor  # um/s 

        dissol = dissol  # um/s

        boundary_cond = nucl / (growth + eps) # num/um or num/um/m**3
        f_aug = np.concatenate(([boundary_cond]*2, csd, [csd[-1]]))

        # Flux source terms
        f_diff = np.diff(f_aug)
        f_diff[f_diff == 0] = eps  # avoid division by zero for theta

        if growth > 0:
            # theta = f_diff[:-1] / (f_diff[1:] + eps*10)
            # theta = f_diff[:-1] / (f_diff[1:] + eps)
            theta = f_diff[:-1] / f_diff[1:]
        else:
            # theta = f_diff[1:] / (f_diff[:-1] + eps*10)
            # theta = f_diff[:-1] / (f_diff[1:] + eps)
            theta = f_diff[:-1] / f_diff[1:]
        # Van-Leer limiter
        limiter = np.zeros_like(f_diff)
        limiter[:-1] = (np.abs(theta) + theta) / (1 + np.abs(theta))

        growth_term = growth * (f_aug[1:-1] + 0.5 * f_diff[1:] * limiter[:-1])
        dissol_term = dissol * (f_aug[2:] - 0.5 * f_diff[1:] * limiter[1:])

        flux = growth_term + dissol_term

        if output == 'flux':
            return flux  # TODO: isn't it necessary to divide by dx?
        elif 'dstates':
            dcsd_dt = -np.diff(flux) / self.dx

            # Material bce in kg_API/s --> G in um, mu_2 in m**2 (or m**2/m**3)
            mass_transfer = rho_cry * kv_cry * (
                3*(growth + dissol)*mu_2 + nucl*self.rad**3) * (1e-6)
            return dcsd_dt, np.array(mass_transfer)
        
    def unit_jacobians(self, time, states, sens, params, fy, v_vector):
        if sens is not None:
            jac_states = self.jac_states_fun(time, states, params)
            jac_params = self.jac_params_fun(time, states, params)

            dsens_dt = np.dot(jac_states, sens) + jac_params

            if not isinstance(dsens_dt, np.ndarray):
                dsens_dt = dsens_dt._value

            return dsens_dt
        elif v_vector is not None:
            _, jac_v = self.jac_states_fun(time, states, params)(v_vector)

            return jac_v
        else:
            jac_states = self.jac_states_fun(time, states, params)

            if not isinstance(jac_states, np.ndarray):
                jac_states = jac_states._value

            return jac_states

    def jac_states_numerical(self, time, states, params, return_only=True):
        if return_only:
            return self.jac_states_vals
        else:
            def wrap_states(st): return self.unit_model(time, st, params)

            abstol = self.sundials_opt['atol']
            reltol = self.sundials_opt['rtol']
            jac_states = numerical_jac_central(wrap_states, states,
                                               dx=dx_jac_x,
                                               abs_tol=abstol, rel_tol=reltol)

            return jac_states

    def jac_params_numerical(self, time, states, params):
        def wrap_params(theta): return self.unit_model(time, states, theta)

        abstol = self.sundials_opt['atol']
        reltol = self.sundials_opt['rtol']
        p_bar = self.sundials_opt['pbar']

        dp = np.abs(p_bar) * np.sqrt(max(reltol, eps))

        jac_params = numerical_jac_central(wrap_params, params,
                                           dx=dp,
                                           abs_tol=abstol, rel_tol=reltol)

        return jac_params
    
    def rhs_sensitivity(self, time, states, sens, params):

        jac_params_vals = self.jac_params_fn(time, states, params)

        jac_states_vals = self.jac_states_fn(time, states, params,
                                             return_only=False)

        rhs_sens = np.dot(jac_states_vals, sens) + jac_params_vals

        self.jac_states_vals = jac_states_vals

        return rhs_sens

    def set_ode_problem(self, eval_sens, states_init, params_mergd,
                        jacv_prod):
        if eval_sens:
            problem = Explicit_Problem(self.unit_model, states_init,
                                       t0=self.elapsed_time,
                                       p0=params_mergd)

            if self.jac_type == 'finite_diff':
                self.jac_states_fn = self.jac_states_numerical
                self.jac_params_fn = self.jac_params_numerical

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type == 'AD':
                self.jac_states_fn = self.jac_states_ad
                self.jac_params_fn = self.jac_params_ad

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type == 'analytical':
                self.jac_states_fn = self.jac_states
                self.jac_params_fn = self.jac_params

                problem.jac = self.jac_states_fn
                problem.rhs_sens = self.rhs_sensitivity

            elif self.jac_type is None:
                pass
            else:
                raise NameError("Bad string value for the 'jac_type' argument")

        else:
            if self.state_event_list is None:
                def model(time, states, params=params_mergd):
                    return self.unit_model(time, states, params)

                problem = Explicit_Problem(model, states_init,
                                           t0=self.elapsed_time)
            else:
                sw0 = [True] * len(self.state_event_list)
                def model(time, states, sw=None):
                    return self.unit_model(time, states, params_mergd, sw)

                problem = Explicit_Problem(model, states_init,
                                           t0=self.elapsed_time, sw0=sw0)

            # ----- Jacobian callables
            if self.method == 'moments':
                # w.r.t. states
                # problem.jac = lambda time, states: \
                #     self.unit_jacobians(time, states, None, params_mergd,
                #                         None, None)

                pass

            elif self.method == 'fvm':
                # J*v product (AD, slower than the one used by SUNDIALS)
                if jacv_prod:
                    problem.jacv = lambda time, states, fy, v: \
                        self.unit_jacobians(time, states, None, params_mergd,
                                            fy, v)

        return problem
    def solve_unit(self, runtime=None, time_grid=None,
                   eval_sens=False,
                   jac_v_prod=False, verbose=True, test=False,
                   sundials_opts=None, any_event=True):
        """
        runtime : float (default = None)
            Value for the total unit runtime
        time_grid : list of float (optional, dafault = None)
            Optional list of time values for the integrator to use
            during simulation
        eval_sens : bool (optional, default = False)
            Boolean value indicating whether the parametric
            sensitivity system will be included during simulation.
            Must be True to access sensitivity information.
        jac_v_prod :
            TODO
        verbose : bool (optional, default = True)
            Boolean value indicating whether the simulator will
            output run statistics after simulation is complete.
            Use True if you want to see the number of function
            evaluations and wall-clock runtime for the unit.
        test :
            TODO
        sundials_opts :
            TODO
        any_event :
            TODO
        """

        if self.__class__.__name__ != 'BatchRC':
            if self.method == 'moments':
                pass  # TODO: MSMPR MoM should be addressed?
            else:
                x_distr = getattr(self.Solid_1, 'x_distrib', [])
                self.states_in_dict['Inlet']['distrib'] = len(x_distr)

        self.CrystKinetics.target_idx = self.target_ind

        # ---------- Solid phase states
        if 'vol' in self.states_uo:
            if self.method == 'moments':
                init_solid = self.Solid_1.moments
                # exp = np.arange(0, self.Solid_1.num_mom) # TODO: problematic line for seeded crystallization.
                # init_solid = init_solid * (1e6)**exp

            elif self.method == '1D-FVM':
                x_grid = self.Solid_1.x_distrib
                init_solid = self.Solid_1.distrib * self.scale

        else:
            if self.method == 'moments':
                init_solid = self.Slurry.moments
                # exp = np.arange(0, self.Solid_1.num_mom) # TODO
                # init_solid = init_solid * (1e6)**exp

            elif self.method == '1D-FVM':
                x_grid = self.Slurry.x_distrib
                init_solid = self.Slurry.distrib * self.scale

        self.dx = self.Slurry.dx
        self.x_grid = self.Slurry.x_distrib

        # ---------- Liquid phase states
        init_liquid = self.Liquid_1.mass_conc.copy()

        self.num_species = len(init_liquid)

        self.len_states = [self.num_distr, self.num_species]  # TODO: not neces

        if 'vol' in self.states_uo:  # Batch or semibatch
            vol_init = self.Slurry.getTotalVol()
            init_susp = np.append(init_liquid, vol_init)

            self.len_states.append(1)
        else:
            init_susp = init_liquid

        if self.reset_states:
            self.reset()

        # ---------- Read time
        if runtime is not None:
            final_time = runtime + self.elapsed_time

        if time_grid is not None:
            final_time = time_grid[-1]

        if self.scale_flag:
            self.scale_flag = False # TODO WHY???

        states_init = np.append(init_solid, init_susp)

        if self.vol_tank is None:
            if isinstance(self, SemibatchRC):
                time_vec = np.linspace(self.elapsed_time, final_time)
                vol_flow = self.get_inputs(time_vec)['Inlet']['vol_flow']

                self.vol_tank = trapezoidal_rule(time_vec, vol_flow)

            else:
                self.vol_tank = self.Slurry.vol

        self.diam_tank = (4/np.pi * self.vol_tank)**(1/3) # TODO ensure redefinition is fine Z
        self.area_base = np.pi/4 * self.diam_tank**2
        self.vol_tank *= 1 / self.vol_offset

        if 'temp_ht' in self.states_uo:

            if len(self.profiles_runs) == 0:
                temp_ht = self.Liquid_1.temp
            else:
                temp_ht = self.profiles_runs[-1]['temp_ht'][-1]

            states_init = np.concatenate(
                (states_init, [self.Liquid_1.temp, temp_ht]))

            self.len_states += [1, 1]
        elif 'temp' in self.states_uo:
            states_init = np.append(states_init, self.Liquid_1.temp)
            self.len_states += [1]

        merged_params = self.CrystKinetics.concat_params()[self.mask_params]

        # ---------- Create problem
        problem = self.set_ode_problem(eval_sens, states_init,
                                       merged_params, jac_v_prod)

        self.derivatives = problem.rhs(self.elapsed_time, states_init,
                                       merged_params)

        if len(self.state_event_list) > 0:
            def new_handle(solver, info):
                return handle_events(solver, info, self.state_event_list,
                                     any_event=any_event)

            problem.state_events = self._eval_state_events
            problem.handle_event = new_handle

        # ---------- Set solver
        # General
        solver = CVode(problem)
        solver.iter = 'Newton'
        solver.discr = 'BDF'

        if sundials_opts is not None:
            for name, val in sundials_opts.items():
                setattr(solver, name, val)

                if name == 'time_limit':
                    solver.report_continuously = True

        self.sundials_opt = solver.get_options()

        if eval_sens:
            solver.sensmethod = 'SIMULTANEOUS'
            solver.suppress_sens = False
            solver.report_continuously = True

        if self.method == '1D-FVM':
            solver.linear_solver = 'SPGMR'  # large, sparse systems

        if not verbose:
            solver.verbosity = 50

        # ---------- Solve model
        time, states = solver.simulate(final_time, ncp_list=time_grid)

        self.retrieve_results(time, states)

        # ---------- Organize sensitivity
        if eval_sens:
            sensit = []
            for elem in solver.p_sol:
                sens = np.array(elem)
                sens[0] = 0  # correct NaN's at t = 0 for sensitivities
                sensit.append(sens)

            self.sensit = sensit

            return time, states, sensit
        else:
            return time, states

    def flatten_states(self):
        out = flatten_states(self.profiles_runs)

        return out




class SemibatchRC(_BaseReactiveCryst):
    pass