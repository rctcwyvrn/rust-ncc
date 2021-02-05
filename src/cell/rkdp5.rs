#![allow(unused)]
use crate::cell::{chemistry::RacRandState, states::CoreState};
use crate::interactions::Interactions;
use crate::math::{max_f64, min_f64};
use crate::parameters::{Parameters, WorldParameters};

type CellDynamicsFn = fn(
    talkative: bool,
    dt: f64,
    tstep: u32,
    int_step: i32,
    state: &CoreState,
    rac_random_state: &RacRandState,
    interactions: &Interactions,
    world_parameters: &WorldParameters,
    parameters: &Parameters,
) -> CoreState;

const C: [f64; 7] =
    [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
// A0s are all zeros
const A1: f64 = 1.0 / 5.0;
const A2: [f64; 2] = [3.0 / 40.0, 9.0 / 40.0];
const A3: [f64; 3] = [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
const A4: [f64; 4] = [
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
];
const A5: [f64; 5] = [
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
];
const A6: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];
const B: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];
const B_HAT: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];
const INV_QP1: f64 = 1.0 / 5.0; // inverse (max of p and p_hat) + 1, see explanation for equation 4.12 in HNW vol1
const FAC: f64 = 0.8; // safety factor, approximately 0.38^QP1, see explanation for equation 4.12 in HWN vol1
const FAC_MAX: f64 = (5.0 - 1.5) / 2.0; // see explanation for equation 4.12 in HWN vol1

pub struct AuxArgs {
    pub max_iters: u32,
    pub atol: f64,
    pub rtol: f64,
    pub init_h_factor: Option<f64>,
}

pub struct SolverArgs {
    f: fn(dt: f64, state: &CoreState, parameters: &Parameters) -> CoreState,
    init_state: CoreState,
    t0: f64,
    t1: f64,
}

pub struct Solution {
    pub y: Result<CoreState, String>,
    pub num_rejections: u32,
    pub num_iters: u32,
}

pub struct Ks {
    k0: CoreState,
    k1: CoreState,
    k2: CoreState,
    k3: CoreState,
    k4: CoreState,
    k5: CoreState,
    k6: CoreState,
}

impl Ks {
    fn calc(
        talkative: bool,
        tstep: u32,
        int_step: i32,
        f: CellDynamicsFn,
        h: f64,
        init_state: CoreState,
        rand_state: &RacRandState,
        inter_state: &Interactions,
        world_parameters: &WorldParameters,
        parameters: &Parameters,
    ) -> Ks {
        // since C[0] = 0.0, the function evaluated at that point will return 0
        let k0 = f(
            talkative,
            h as f64,
            tstep,
            int_step,
            &init_state,
            rand_state,
            inter_state,
            world_parameters,
            parameters,
        );

        let k1 = {
            let kp = init_state + h * A1 * k0;
            f(
                talkative,
                h as f64,
                tstep,
                int_step,
                &kp,
                rand_state,
                inter_state,
                world_parameters,
                parameters,
            )
        };

        let k2 = {
            let kp = init_state + h * (A2[0] * k0 + A2[1] * k1);
            f(
                talkative,
                h as f64,
                tstep,
                int_step,
                &kp,
                rand_state,
                inter_state,
                world_parameters,
                parameters,
            )
        };

        let k3 = {
            let kp = init_state + h * (A3[0] * k0 + A3[1] * k1 + A3[2] * k2);
            f(
                talkative,
                h as f64,
                tstep,
                int_step,
                &kp,
                rand_state,
                inter_state,
                world_parameters,
                parameters,
            )
        };

        let k4 = {
            let kp = init_state
                + h * (A4[0] * k0 + A4[1] * k1 + A4[2] * k2 + A4[3] * k3);
            f(
                talkative,
                h as f64,
                tstep,
                int_step,
                &kp,
                rand_state,
                inter_state,
                world_parameters,
                parameters,
            )
        };

        let k5 = {
            let kp = init_state
                + h * (A5[0] * k0
                    + A5[1] * k1
                    + A5[2] * k2
                    + A5[3] * k3
                    + A5[4] * k4);
            f(
                talkative,
                h as f64,
                tstep,
                int_step,
                &kp,
                rand_state,
                inter_state,
                world_parameters,
                parameters,
            )
        };

        let k6 = {
            let kp = init_state
                + h * (A6[0] * k0
                    + A6[1] * k1
                    + A6[2] * k2
                    + A6[3] * k3
                    + A6[4] * k4
                    + A6[5] * k5);
            f(
                talkative,
                h as f64,
                tstep,
                int_step,
                &kp,
                rand_state,
                inter_state,
                world_parameters,
                parameters,
            )
        };

        Ks {
            k0,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
        }
    }
}

pub fn integrator(
    talkative: bool,
    tstep: u32,
    mut dt: f64,
    f: CellDynamicsFn,
    init_state: &CoreState,
    rand_state: &RacRandState,
    inter_state: &Interactions,
    world_parameters: &WorldParameters,
    parameters: &Parameters,
    mut aux_args: AuxArgs,
) -> Solution {
    let mut y0 = *init_state;

    let AuxArgs {
        max_iters,
        atol,
        rtol,
        init_h_factor,
    } = aux_args;

    let mut h = if let Some(h_factor) = init_h_factor {
        h_factor * dt
    } else {
        0.1 * dt
    };

    let mut num_iters = 0_u32;
    let mut fac_max = FAC_MAX;
    let mut last_iter = false;
    let mut num_rejections: u32 = 0;
    let mut int_step = 0;

    while num_iters < max_iters {
        let Ks {
            k0,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
        } = Ks::calc(
            talkative,
            tstep,
            int_step,
            f,
            h,
            y0,
            rand_state,
            inter_state,
            world_parameters,
            parameters,
        );

        let y1 = y0
            + h * (B[0] * k0
                + B[1] * k1
                + B[2] * k2
                + B[3] * k3
                + B[4] * k4
                + B[5] * k5
                + B[6] * k6);

        if last_iter {
            assert!((h - dt).abs() < f64::EPSILON);
            return Solution {
                y: Ok(y1),
                num_rejections,
                num_iters,
            };
        }

        let y1_hat = y0
            + h * (B_HAT[0] * k0
                + B_HAT[1] * k1
                + B_HAT[2] * k2
                + B_HAT[3] * k3
                + B_HAT[4] * k4
                + B_HAT[5] * k5
                + B_HAT[6] * k6);

        // Equations 4.10, 4.11, Hairer,Wanner&Norsett Solving ODEs Vol. 1
        let sc = y0.abs().max(&y1.abs()).scalar_mul(rtol).scalar_add(atol);
        let error = ((y1 - y1_hat).powi(2) / sc).average().sqrt();
        let mut h_new = h * min_f64(fac_max, FAC * (1.0 / error).powf(INV_QP1));

        // see explanation for equation 4.13 in HNW vol1
        if error <= 1.0 {
            fac_max = FAC_MAX;
            y0 = y1;
            if h + h_new > dt {
                h_new = dt - h;
                last_iter = true;
            };
            dt -= h;
            h = h_new;
        } else {
            int_step -= 1;
            fac_max = 1.0;
            num_rejections += 1;
            h = h_new;
        }

        num_iters += 1;
        int_step += 1;
    }

    Solution {
        num_rejections,
        num_iters,
        y: Err("Too many iterations!".to_string()),
    }
}
