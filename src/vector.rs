use crate::*;
use std::{
    array::TryFromSliceError,
    convert::{TryFrom, TryInto},
    usize,
};

pub type Vector<T, const N: usize> = Matrix<T, N, 1>;

impl<T, const N: usize> Vector<T, N> {
    pub const fn new(values: [T; N]) -> Self {
        Self([values])
    }
}

impl<T: Numeric, const N: usize> TryFrom<&[T]> for Vector<T, N> {
    type Error = TryFromSliceError;
    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        Ok(Self([value.try_into()?]))
    }
}

impl<T> From<T> for Vector<T, 1> {
    fn from(value: T) -> Self {
        Self([[value]])
    }
}

impl<T> From<(T,)> for Vector<T, 1> {
    fn from(value: (T,)) -> Self {
        Self([[value.0]])
    }
}

impl<T> From<(T, T)> for Vector<T, 2> {
    fn from(value: (T, T)) -> Self {
        Self([[value.0, value.1]])
    }
}

impl<T: Copy> From<(T, T, T)> for Vector<T, 3> {
    fn from(value: (T, T, T)) -> Self {
        Self([[value.0, value.1, value.2]])
    }
}

impl<T: Copy> From<(T, T, T, T)> for Vector<T, 4> {
    fn from(value: (T, T, T, T)) -> Vector<T, 4> {
        Self([[value.0, value.1, value.2, value.3]])
    }
}

impl<T: Copy> From<Vector<T, 1>> for (T,) {
    fn from(value: Vector<T, 1>) -> (T,) {
        (value.0[0][0],)
    }
}

impl<T: Copy> From<Vector<T, 2>> for (T, T) {
    fn from(value: Vector<T, 2>) -> (T, T) {
        (value.0[0][0], value.0[0][1])
    }
}

impl<T: Copy> From<Vector<T, 3>> for (T, T, T) {
    fn from(value: Vector<T, 3>) -> (T, T, T) {
        (value.0[0][0], value.0[0][1], value.0[0][2])
    }
}

impl<T: Copy> From<Vector<T, 4>> for (T, T, T, T) {
    fn from(value: Vector<T, 4>) -> (T, T, T, T) {
        (value.0[0][0], value.0[0][1], value.0[0][2], value.0[0][3])
    }
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn as_array(self) -> [T; N] {
        self.0[0]
    }

    /// Dot product of `self` and `other`
    pub fn dot(self, other: Self) -> T {
        let mut total = T::ZERO;
        for i in 0..N {
            total = total + self.0[0][i] * other.0[0][i];
        }
        total
    }

    /// Returns a `Vector<T, 2>` with x and y components.
    /// If this `Vector` has fewer than 2 components then the extra
    /// components are set to 0.
    pub fn xy(&self) -> Vector<T, 2> {
        Vector::new([self.0[0][0], if N > 1 { self.0[0][1] } else { T::ZERO }])
    }

    /// Returns a `Vector<T, 3>` with x, y, and z components.
    /// If this `Vector` has fewer than 3 components then the extra
    /// components are set to 0.
    pub fn xyz(&self) -> Vector<T, 3> {
        Vector::new([
            self.0[0][0],
            if N > 1 { self.0[0][1] } else { T::ZERO },
            if N > 2 { self.0[0][2] } else { T::ZERO },
        ])
    }

    /// Returns a `Vector<T, 4>` with x, y, z, w components.
    /// If this `Vector` has fewer than 4 components then the extra
    /// components are set to 0.
    pub fn xyzw(&self) -> Vector<T, 4> {
        Vector::new([
            self.0[0][0],
            if N > 1 { self.0[0][1] } else { T::ZERO },
            if N > 2 { self.0[0][2] } else { T::ZERO },
            if N > 3 { self.0[0][2] } else { T::ZERO },
        ])
    }

    pub fn zxy(&self) -> Vector<T, 3> {
        Vector::new([
            if N > 2 { self.0[0][2] } else { T::ZERO },
            self.0[0][0],
            if N > 1 { self.0[0][1] } else { T::ZERO },
        ])
    }
}

impl<T: Numeric> Vector<T, 1> {
    pub fn extend(self, y: T) -> Vector<T, 2> {
        Vector::new([self.0[0][0], y])
    }
}

impl<T: Numeric> Vector<T, 2> {
    pub fn extend(self, z: T) -> Vector<T, 3> {
        Vector::new([self.0[0][0], self.0[0][1], z])
    }
}

impl<T: Numeric> Vector<T, 3> {
    pub fn extend(self, w: T) -> Vector<T, 4> {
        Vector::new([self.0[0][0], self.0[0][1], self.0[0][2], w])
    }
}

impl<T: Numeric + NumericSqrt, const N: usize> Vector<T, N> {
    /// Calculates the length of this `Vector`
    pub fn length(self) -> T {
        self.dot(self).numeric_sqrt()
    }

    /// Returns a new `Vector` with a length of 1.0
    pub fn normalized(self) -> Self {
        self / self.dot(self).numeric_sqrt()
    }
}

impl<T: Numeric> Vector<T, 1> {
    pub const UNIT_X: Self = {
        let mut v = Self::ZERO;
        v.0[0][0] = T::ONE;
        v
    };
}

impl<T: Numeric> Vector<T, 2> {
    pub const UNIT_X: Self = {
        let mut v = Self::ZERO;
        v.0[0][0] = T::ONE;
        v
    };
    pub const UNIT_Y: Self = {
        let mut v = Self::ZERO;
        v.0[0][1] = T::ONE;
        v
    };
}

impl<T: Numeric> Vector<T, 3> {
    pub const UNIT_X: Self = {
        let mut v = Self::ZERO;
        v.0[0][0] = T::ONE;
        v
    };
    pub const UNIT_Y: Self = {
        let mut v = Self::ZERO;
        v.0[0][1] = T::ONE;
        v
    };
    pub const UNIT_Z: Self = {
        let mut v = Self::ZERO;
        v.0[0][2] = T::ONE;
        v
    };
}

impl<T: Numeric> Vector<T, 4> {
    pub const UNIT_X: Self = {
        let mut v = Self::ZERO;
        v.0[0][0] = T::ONE;
        v
    };
    pub const UNIT_Y: Self = {
        let mut v = Self::ZERO;
        v.0[0][1] = T::ONE;
        v
    };
    pub const UNIT_Z: Self = {
        let mut v = Self::ZERO;
        v.0[0][2] = T::ONE;
        v
    };
    pub const UNIT_W: Self = {
        let mut v = Self::ZERO;
        v.0[0][3] = T::ONE;
        v
    };
}

impl<T: Numeric> Vector<T, 3> {
    /// Produces a `Vector<T, 3>` perpendicular to `self` and `other`.
    /// Only applicable to 3-dimensional `Vector`s.
    pub fn cross(self, other: Self) -> Self {
        (self.zxy().mul_by_component(other) - self.mul_by_component(other.zxy())).zxy()
    }
}
