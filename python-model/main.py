import experiment_templates as ets
import writer as fw

TIME_IN_HOURS = 0.01
NUM_CELLS = 2
BOX_WIDTH = 2
BOX_HEIGHT = 1
fw.remake_write_file(int((TIME_IN_HOURS * 3600) / 2), NUM_CELLS)

closeness_dist_squared_criteria = 0.5e-6 ** 2

parameter_dict = dict(
    [
        ("num_nodes", 16),
        ("init_cell_radius", 20e-6),
        ("C_total", 2.5e6),
        ("H_total", 1e6),
        ("init_rgtpase_cytosol_frac", 0.8),
        ("init_rgtpase_membrane_active_frac", 0.1),
        ("init_rgtpase_membrane_inactive_frac", 0.1),
        ("diffusion_const", 0.1e-12),
        ("kgdi_multiplier", 1),
        ("kdgdi_multiplier", 1),
        ("kgtp_rac_multiplier", 1.0),
        ("kgtp_rac_autoact_multiplier", 200),
        ("kdgtp_rac_multiplier", 5.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 1000),
        ("threshold_rac_activity_multiplier", 0.4),
        ("kgtp_rho_multiplier", 10.0),
        ("kgtp_rho_autoact_multiplier", 100),
        ("kdgtp_rho_multiplier", 2.5),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 1000.0),
        ("threshold_rho_activity_multiplier", 0.4),
        ("hill_exponent", 3),
        ("tension_mediated_rac_inhibition_half_strain", 0.05),
        ("tension_mediated_rac_inhibition_magnitude", 1.0),
        ("coa_sensing_dist_at_value", 110e-6),
        ("coa_sensing_value_at_dist", 0.5),
        ("interaction_factor_migr_bdry_contact", 30.0),
        ("closeness_dist_squared_criteria", closeness_dist_squared_criteria),
        ("length_3D_dimension", 10e-6),
        ("stiffness_edge", 5000),
        ("stiffness_cytoplasmic", 1e-5),
        ("eta", 1e5),
        ("max_force_rac", 10e3),
        ("force_rho_multiplier", 0.2),
        ("force_adh_const", 0.0),
        ("skip_dynamics", False),
        ("randomization_scheme", "m"),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 10.0),
        ("randomization_node_percentage", 0.25),
        ("randomization_type", "r"),
        ("coa_intersection_exponent", 2.0),
        ("strain_calculation_type", 0),
        ("max_coa_signal", -1.0),
    ]
)

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1
allowed_drift_before_geometry_recalc = 20.0

remake_animation = False
remake_graphs = False
do_final_analysis = True

default_cil = 60.0
integration_params = {"rtol": 1e-4}

base_output_dir = "/python-model/output"

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 12.0),
        ("kgtp_rho_multiplier", 14.0),
        ("kdgtp_rac_multiplier", 4.0),
        ("kdgtp_rho_multiplier", 30.0),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 195.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 200.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.1),
        ("tension_mediated_rac_inhibition_magnitude", 40.0),
        ("max_force_rac", 3000.0),
        ("eta", 2.9 * 10000.0),
        ("stiffness_edge", 8000.0),
        ("randomization_time_mean", 20.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 12.0),
        ("randomization_node_percentage", 0.25),
    ]
)

sub_experiment_number = 0

coa_dict = {
    49: 8.0,
    36: 9.0,
    25: 12.0,
    16: 14.0,
    9: 16.0,
    4: 24.0,
    2: 24.0,
    1: 24.0}

if __name__ == "__main__":
    uniform_initial_polarization = False
    parameter_dict.update(
        [
            ("randomization_scheme", "m"),
        ]
    )
    standard_rps = [
        ("randomization_scheme", "m"),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 10.0),
        ("randomization_node_percentage", 0.25),
    ]
    parameter_dict.update(standard_rps)

    ets.rust_comparison_test(
        sub_experiment_number,
        parameter_dict,
        uniform_initial_polarization=False,
        no_randomization=True,
        total_time_in_hours=TIME_IN_HOURS,
        timestep_length=2,
        integration_params=integration_params,
        allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
        default_coa=24.0,
        default_cil=60.0,
        num_experiment_repeats=1,
        box_width=BOX_WIDTH,
        box_height=BOX_HEIGHT,
        num_cells=NUM_CELLS,
        biased_rgtpase_distrib_defn_dict={
            "default": ["biased nodes", [0, 1, 2, 3], 0.1]
        },
        justify_parameters=True,
    )
