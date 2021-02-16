// Copyright © 2020 Brian Merchant.
//
// Licensed under the Apache License, Version 2.inner <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.inner> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::math::modulo_f64;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Radians {
    inner: f64,
}

pub const RAD_2PI: Radians = Radians { inner: 2.0 * PI };
pub const RAD_EPS: Radians = Radians { inner: 1e-4 * PI };

impl From<f64> for Radians {
    fn from(val: f64) -> Self {
        Radians {
            inner: modulo_f64(val, RAD_2PI.inner),
        }
    }
}

impl PartialEq for Radians {
    fn eq(&self, other: &Self) -> bool {
        (self.inner - other.inner).abs() < RAD_EPS.inner
    }
}

impl PartialOrd for Radians {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Radians::from(self.inner)
            .inner
            .partial_cmp(&Radians::from(other.inner).inner)
    }
}

/// Addition modulo `2*PI`.
impl Add for Radians {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Radians::from(self.inner + other.inner)
    }
}

/// Subtraction modulo `2*PI`.
impl Sub for Radians {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Radians::from(self.inner - other.inner)
    }
}

impl Mul<f64> for Radians {
    type Output = Radians;

    fn mul(self, rhs: f64) -> Self::Output {
        Radians::from(self.inner * rhs)
    }
}

impl Mul<Radians> for f64 {
    type Output = Radians;

    fn mul(self, rhs: Radians) -> Self::Output {
        Radians::from(self * rhs.inner)
    }
}

impl Radians {
    /// Helper function for `Radians::between`.
    fn _between(&self, t0: Radians, t1: Radians) -> bool {
        t0 < *self && *self < t1
    }

    /// Calculates if `self` is between `t0` and `t1`, where `t1` is assumed to be counter-clockwise after `t0`.
    pub fn between(&self, t0: Radians, t1: Radians) -> bool {
        if t0 - t1 < RAD_EPS {
            false
        } else if t1 - *self < RAD_EPS || t0 - *self < RAD_EPS {
            true
        } else if t0 < t1 {
            self._between(t0, t1)
        } else {
            !self._between(t1, t0)
        }
    }
}
