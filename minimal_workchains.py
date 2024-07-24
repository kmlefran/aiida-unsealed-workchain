from aiida.plugins import CalculationFactory
from aiida.engine import BaseRestartWorkChain, while_, process_handler, ProcessHandlerReport, WorkChain, calcfunction
from aiida.common import AttributeDict
from aiida.orm import Str, Dict, Code


GaussianCalculation = CalculationFactory("aimall.gaussianwfx")

@calcfunction
def geom_from_dict(param_dict):
    symbols = param_dict.get_dict()['atomnos']
    geom = param_dict.get_dict()['atomcoords'][-1]
    structure_str = ''
    for x in range(len(symbols)):
        if x != len(symbols) - 1:
            structure_str = (
                structure_str + 
                f"{symbols[x]}  {geom[x][0]:.6f}    {geom[x][1]:.6f}    {geom[x][2]:.6f}\n"
            )
        else:
            structure_str = (
                structure_str + 
                f"{symbols[x]}  {geom[x][0]:.6f}    {geom[x][1]:.6f}    {geom[x][2]:.6f}"
            )
    return Str(structure_str)

@calcfunction
def get_diffuse_param_dict(g16_params, basis_set_string):
    diffuse_param_dict = g16_params.get_dict()
    diffuse_param_dict['basis_set'] = 'gen'
    if not diffuse_param_dict['input_parameters']:
        diffuse_param_dict['input_parameters'] = {basis_set_string.value : None}
    else:
        diffuse_param_dict['input_parameters'] = {''.join(list(diffuse_param_dict['input_parameters'].keys())) + '\n\n' + basis_set_string.value : None}
    return Dict(diffuse_param_dict)

@calcfunction
def get_implicit_solv_inputs(g16_params, solvent):
    implicit_solv_param_dict = g16_params.get_dict()
    implicit_solv_param_dict['route_parameters']['scrf'] = {'iefpcm' : None, 'solvent' : solvent.value}
    return Dict(implicit_solv_param_dict)

@calcfunction
def get_implicit_solv_and_diffuse_inputs(g16_params, basis_set_string, solvent):
    hybrid_param_dict = g16_params.get_dict()
    hybrid_param_dict['basis_set'] = 'gen'
    if not hybrid_param_dict['input_parameters']: #if there is no modredundant info in the 'input_parameters'
        hybrid_param_dict['input_parameters'] = {basis_set_string.value : None}
    else:
        hybrid_param_dict['input_parameters'] = {''.join(list(hybrid_param_dict['input_parameters'].keys())) + '\n\n' + basis_set_string.value : None}
        #if the input parameters contains modredundant info, extract the string, add a blank line in between, and then append basis set information"
    hybrid_param_dict['route_parameters']['scrf'] = {'iefpcm' : None, 'solvent' : solvent.value}
    return Dict(hybrid_param_dict)

class GaussianBaseRestartWorkChain(BaseRestartWorkChain): #stolen from aiida-gaussian plugin and modified slightly
    """Workchain to run a Gaussian calculation with automated error handling and restarts."""

    _process_class = GaussianCalculation

    @classmethod
    def define(cls, spec):

        super().define(spec)
        spec.expose_inputs(GaussianCalculation, namespace="gaussian")

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            350,
            "ERROR_UNRECOVERABLE_SCF_FAILURE",
            message="The calculation failed with an unrecoverable SCF convergence error.",
        )

        spec.exit_code(
            399,
            "ERROR_UNRECOVERABLE_TERMINATION",
            message="The calculation failed with an unrecoverable error.",
        )

    def setup(self):
        """Call the `setup` and create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to
        submit the calculations in the internal loop.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(
            self.exposed_inputs(GaussianCalculation, "gaussian")
        )

    @process_handler(
        priority=0,
        exit_codes=[GaussianCalculation.exit_codes.ERROR_NO_NORMAL_TERMINATION, GaussianCalculation.exit_codes.ERROR_TERMINATION, GaussianCalculation.exit_codes.ERROR_OUTPUT_PARSING],
    )
    def handle_misc_failure(self, node):
        """
        By default, the BaseRestartWorkChain restarts any unhandled error once
        Disable this feature for the exit_code that corresponds to out-of-time
        """
        param_dict = node.outputs.output_parameters.get_dict()
        
        symbols = param_dict['atomnos']
        geom = param_dict['atomcoords'][-1]
        structure_str = ''
        for x in range(len(symbols)):
            if x != len(symbols) - 1:
                structure_str = (
                    structure_str + 
                    f"{symbols[x]}  {geom[x][0]:.6f}    {geom[x][1]:.6f}    {geom[x][2]:.6f}\n"
                )
            else:
                structure_str = (
                    structure_str + 
                    f"{symbols[x]}  {geom[x][0]:.6f}    {geom[x][1]:.6f}    {geom[x][2]:.6f}"
                )
        self.ctx.inputs.structure_str = Str(structure_str)
        return ProcessHandlerReport()

    def results(self):
        """Overload the method such that each dynamic output of GaussianCalculation is set."""
        node = self.ctx.children[self.ctx.iteration - 1]

        # We check the `is_finished` attribute of the work chain and not the successfulness of the last process
        # because the error handlers in the last iteration can have qualified a "failed" process as satisfactory
        # for the outcome of the work chain and so have marked it as `is_finished=True`.
        max_iterations = self.inputs.max_iterations.value  # type: ignore[union-attr]
        if not self.ctx.is_finished and self.ctx.iteration >= max_iterations:
            self.report(
                f"reached the maximum number of iterations {max_iterations}: "
                f"last ran {self.ctx.process_name}<{node.pk}>"
            )
            return (
                self.exit_codes.ERROR_MAXIMUM_ITERATIONS_EXCEEDED
            )  # pylint: disable=no-member

        self.report(f"The work chain completed after {self.ctx.iteration} iterations")

        self.out_many({key: node.outputs[key] for key in node.outputs})

        return None

class DiffuseImplicitHybridWorkChain(WorkChain):
    """workchain to run 3 calculations on a geometry given starting structure -- one with implcit solvation (IEFPCM), one with 
    one iwth diffuse functionals, one hybrid
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('g16_base_params', valid_type=Dict, required=True)
        spec.input('input_structure', valid_type=Str, required=True)
        spec.input('molecule_name', valid_type=Str, required=True)
        spec.input('original_parsed_data', valid_type=Dict, required=True) 
        #Note: the ['metadata'] key contains a non-supported datatype and must be removed prior to passing the cclib dict
        #can be accomplished with:
        #   cclib_dict = cclib_data.__dict__
        #   del cclib_dict['metadata']
        #   alternatively, a make_serializable function is availble in aiida-gaussian
        spec.input('g16_code', valid_type=Code, required=True)
        spec.input('diffuse_basis_set_data', valid_type=Str, required=True)
        spec.input('solvent', valid_type=Str, required=False, default=lambda: Str('water'))
        spec.output('geometry', valid_type=Dict)
        spec.output('output_dict', valid_type=Dict)
        spec.output('free_energy', valid_type=Dict)
        spec.outline(
            cls.run_jobs,
            cls.result
        )
    
    def run_jobs(self):

        #first set up diffuse function calculation
        diffuse_inputs = {
            'gaussian' : {
                'structure_str' : self.inputs.input_structure,
                'parameters' : get_diffuse_param_dict(self.inputs.g16_base_params, self.inputs.diffuse_basis_set_data),
                'code' : self.inputs.g16_code,
                'metadata' : {
                    'options' : {
                        'resources' : {"num_machines": 1, "tot_num_mpiprocs": 4},
                        'max_memory_kb' : int(3200 * 1.25) * 1024,
                        'max_wallclock_seconds' : 604800,
                    }
                }
            }      
        }

        implicit_inputs = {
            'gaussian' : {
                'structure_str' : self.inputs.input_structure,
                'parameters' : get_implicit_solv_inputs(self.inputs.g16_base_params, self.inputs.solvent),
                'code' : self.inputs.g16_code,
                'metadata' : {
                    'options' : {
                        'resources' : {"num_machines": 1, "tot_num_mpiprocs": 4},
                        'max_memory_kb' : int(3200 * 1.25) * 1024,
                        'max_wallclock_seconds' : 604800,
                    }
                }
            }  
        }

        hybrid_inputs = {
            'gaussian' : {
                'structure_str' : self.inputs.input_structure,
                'parameters' : get_implicit_solv_and_diffuse_inputs(self.inputs.g16_base_params, self.inputs.diffuse_basis_set_data, self.inputs.solvent),
                'code' : self.inputs.g16_code,
                'metadata' : {
                    'options' : {
                        'resources' : {"num_machines": 1, "tot_num_mpiprocs": 4},
                        'max_memory_kb' : int(3200 * 1.25) * 1024,
                        'max_wallclock_seconds' : 604800,
                    }
                }
            }  
        }
        diffuse_node = self.submit(GaussianBaseRestartWorkChain, **diffuse_inputs)
        implicit_node = self.submit(GaussianBaseRestartWorkChain, **implicit_inputs)
        hybrid_node = self.submit(GaussianBaseRestartWorkChain, **hybrid_inputs)

        diffuse_node.base.extras.set_many({'molecule_name' : self.inputs.molecule_name, 'theory_level' : 'diffuse_functional'})
        implicit_node.base.extras.set_many({'molecule_name' : self.inputs.molecule_name, 'theory_level' : 'implicit_solvation'})
        hybrid_node.base.extras.set_many({'molecule_name' : self.inputs.molecule_name, 'theory_level' : 'diffuse_functional_and_implicit_solvation'})

        out_dict = {
            'g16.diffuse' : diffuse_node,
            'g16.implicit' : implicit_node,
            'g16.hybrid' : hybrid_node,
        }
        self.to_context(**out_dict)

    def result(self):
        energy_dict = {}
        geom_dict = {}
        output_dict = {}
        for i, (calc_type, node) in enumerate(self.ctx.g16.items()):
            key = self.inputs.molecule_name.value + '_' + calc_type
            param_dict = node.base.links.get_outgoing().get_node_by_label('output_parameters').get_dict()
            energy_dict[key] = param_dict['freeenergy']
            geom_dict[key] = geom_from_dict(param_dict).value
            output_dict[key] = param_dict

        #include the original values (without diffuse functionals or solvation) for comparison
        energy_dict['original'] = self.inputs.original_parsed_data.get_dict()['freeenergy']
        geom_dict['original'] = self.inputs.input_structure.value
        output_dict['original'] = self.inputs.original_parsed_data.get_dict()

        self.out('free_energy', Dict(energy_dict))
        self.out('geometry', Dict(geom_dict))
        self.out('output_dict', Dict(output_dict))