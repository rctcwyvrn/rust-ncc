use crate::math::radians::{Radians, RAD_2PI};
use crate::NVERTS;
use once_cell::sync::Lazy;

pub fn mark_between_angles(
    bounds: (Radians, Radians),
) -> [bool; NVERTS] {
    let mut r = [false; NVERTS];
    let (b0, b1) = bounds;
    for vi in 0..NVERTS {
        let va = (vi as f64 / NVERTS as f64) * RAD_2PI;
        if b0 <= va && va <= b1 {
            r[vi] = true;
        }
    }
    r
}

pub static TOP_HALF: Lazy<[bool; NVERTS]> =
    Lazy::new(|| mark_between_angles((0.0 * RAD_2PI, 0.5 * RAD_2PI)));
pub static BOT_HALF: Lazy<[bool; NVERTS]> =
    Lazy::new(|| mark_between_angles((0.5 * RAD_2PI, RAD_2PI)));
pub const ALL: [bool; NVERTS] = [true; NVERTS];

#[macro_export]
macro_rules! mark_verts {
    ( $( $x:literal ),* ) => {{
        let mut marks = [false; NVERTS];
        $(marks[$x] = true;)*
        marks
    }};
}
