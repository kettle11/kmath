use std::{
    default::Default,
    ops::{Add, Div, Mul, Neg, Sub},
};

pub trait Numeric:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Copy
    + Default
    + Clone
{
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;
    const MIN: Self;

    fn numeric_min(self, other: Self) -> Self;
    fn numeric_max(self, other: Self) -> Self;
}

pub trait NumericFloat: Numeric + NumericSqrt + NumericAbs + Neg<Output = Self> {
    const HALF: Self;
    const TWO: Self;

    fn sin_cos_numeric(self) -> (Self, Self);
    fn tan_numeric(self) -> Self;
    fn is_nan_numeric(self) -> bool;
    fn copysign_numeric(self, sign: Self) -> Self;
}

impl NumericFloat for f32 {
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    fn sin_cos_numeric(self) -> (Self, Self) {
        self.sin_cos()
    }
    fn tan_numeric(self) -> Self {
        self.tan()
    }
    fn is_nan_numeric(self) -> bool {
        self.is_nan()
    }
    fn copysign_numeric(self, sign: Self) -> Self {
        self.copysign(sign)
    }
}

impl NumericFloat for f64 {
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    fn sin_cos_numeric(self) -> (Self, Self) {
        self.sin_cos()
    }
    fn tan_numeric(self) -> Self {
        self.tan()
    }
    fn is_nan_numeric(self) -> bool {
        self.is_nan()
    }
    fn copysign_numeric(self, sign: Self) -> Self {
        self.copysign(sign)
    }
}

pub trait NumericSqrt {
    fn numeric_sqrt(self) -> Self;
}

pub trait NumericAbs {
    fn numeric_abs(self) -> Self;
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericAbs for f32 {
    fn numeric_abs(self) -> Self {
        self.abs()
    }
}

impl NumericSqrt for f32 {
    fn numeric_sqrt(self) -> Self {
        self.sqrt()
    }
}

impl Numeric for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericAbs for f64 {
    fn numeric_abs(self) -> Self {
        self.abs()
    }
}

impl NumericSqrt for f64 {
    fn numeric_sqrt(self) -> Self {
        self.sqrt()
    }
}

impl Numeric for i8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericAbs for i8 {
    fn numeric_abs(self) -> Self {
        self.abs()
    }
}

impl Numeric for i16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericAbs for i16 {
    fn numeric_abs(self) -> Self {
        self.abs()
    }
}

impl Numeric for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericAbs for i32 {
    fn numeric_abs(self) -> Self {
        self.abs()
    }
}

impl Numeric for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericAbs for i64 {
    fn numeric_abs(self) -> Self {
        self.abs()
    }
}

impl Numeric for i128 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericAbs for i128 {
    fn numeric_abs(self) -> Self {
        self.abs()
    }
}

impl Numeric for u8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl Numeric for u16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl Numeric for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl Numeric for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl Numeric for u128 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = Self::MAX;
    const MIN: Self = Self::MIN;
    fn numeric_min(self, other: Self) -> Self {
        self.min(other)
    }
    fn numeric_max(self, other: Self) -> Self {
        self.max(other)
    }
}
