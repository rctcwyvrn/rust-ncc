import experiment_templates as ets
import hardio as fw
import numpy as np

TIME_IN_HOURS = 1.0
CHAR_T = 2.0
CHAR_L = 1e-6
CHAR_F = 1e-9
CHAR_ETA = 2.9 * 10000.0
CHAR_L3D = 10e-6
CHAR_K_MEM_ON = 0.02
CHAR_K_MEM_OFF = 0.15
NUM_CELLS = 2
BOX_WIDTH = 2
BOX_HEIGHT = 1
COA = 24.0
CIL = 60.0
NUM_INT_STEPS = 10
TIME_IN_SECS = TIME_IN_HOURS * 3600.0
NUM_TSTEPS = int(TIME_IN_SECS / CHAR_T)

world_params = dict([
    ("char_t", CHAR_T),
    ("char_l", CHAR_L)
])

params = dict(
    [
        ("nverts", 16),
        ("cell_r", 20e-6),
        ("tot_rac", 2.5e6),
        ("tot_rho", 1e6),
        ("init_act_rgtp", 0.1),
        ("init_inact_rgtp", 0.1),
        ("diffusion_rgtp", 0.1e-12),
        ("kgdi_multiplier", 1),
        ("kdgdi_multiplier", 1),
        ("threshold_rac_activity_multiplier", 0.4),
        ("threshold_rho_activity_multiplier", 0.4),
        ("hill_exponent", 3),
        ("coa_sensing_dist_at_value", 110e-6),
        ("close_criterion", 0.5e-6 ** 2),
        ("stiffness_cyto", 1e-5),
        ("force_rho_multiplier", 0.2),
        ("force_adh_const", 0.0),
        ("randomization_scheme", "m"),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 10.0),
        ("randomization_node_percentage", 0.25),
        ("los_factor", 2.0),
    ]
)

params.update(
    [
        ("kgtp_rac_multiplier", 24.0),
        ("kgtp_rho_multiplier", 28.0),
        ("kdgtp_rac_multiplier", 8.0),
        ("kdgtp_rho_multiplier", 60.0),
        ("kgtp_rac_autoact_multiplier", 500.0),
        ("kgtp_rho_autoact_multiplier", 390.0),
        ("kgtp_rac_on_rho", 400.0),
        ("kdgtp_rho_on_rac", 4000.0),
        ("halfmax_tension_inhib", 0.1),
        ("tension_inhib", 40.0),
        ("max_force_rac", 3000.0),
        ("stiffness_edge", 8000.0),
    ]
)

# coa_dict = {
#     49: 8.0,
#     36: 9.0,
#     25: 12.0,
#     16: 14.0,
#     9: 16.0,
#     4: 24.0,
#     2: 24.0,
#     1: 24.0}

if __name__ == "__main__":
    uniform_initial_polarization = False

    ets.rust_comparison_test(
        NUM_CELLS,
        NUM_TSTEPS,
        NUM_INT_STEPS,
        CIL,
        COA,
        params,
        uniform_initial_polarization=False,
        no_randomization=True,
        tstep_in_secs=2,
        box_width=BOX_WIDTH,
        box_height=BOX_HEIGHT,
        rgtp_distrib_def_dict={
            "default": ["biased nodes", [0, 1, 2, 3], 0.1]
        },
        justify_parameters=True,
    )
