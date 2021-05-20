use crate::*;

/// A circle in 2D, a sphere in 3D.
pub struct Ball<T, const DIMENSIONS: usize> {
    pub center: Vector<T, DIMENSIONS>,
    pub radius: T,
}

impl<T, const DIMENSIONS: usize> Ball<T, DIMENSIONS> {
    pub fn new(center: Vector<T, DIMENSIONS>, radius: T) -> Self {
        Self { center, radius }
    }
}

pub struct Line<T, const DIMENSIONS: usize> {
    pub point: Vector<T, DIMENSIONS>,
    pub direction: Vector<T, DIMENSIONS>,
}

pub struct LineSegment<T, const DIMENSIONS: usize> {
    pub a: Vector<T, DIMENSIONS>,
    pub b: Vector<T, DIMENSIONS>,
}

// Returns magnitude of distance and the point
pub fn closest_point_on_line_segment<T: NumericFloat, const DIMENSIONS: usize>(
    point: Vector<T, DIMENSIONS>,
    line_segment: LineSegment<T, DIMENSIONS>,
) -> Vector<T, DIMENSIONS> {
    let ba = line_segment.b - line_segment.a;
    let pa = point - line_segment.a;
    let h = (ba.dot(pa) / ba.dot(ba))
        .numeric_max(T::ZERO)
        .numeric_min(T::ONE);
    let position = line_segment.a + (ba * h);
    position
}

/// A rectangle in 2D, a rectangular prism in 3D.
pub struct Rect<T, const DIMENSIONS: usize> {
    pub min: Vector<T, DIMENSIONS>,
    pub max: Vector<T, DIMENSIONS>,
}

impl<T: Numeric + PartialOrd, const DIMENSIONS: usize> Rect<T, DIMENSIONS> {
    pub const ZERO: Self = Self {
        min: Vector::<T, DIMENSIONS>::ZERO,
        max: Vector::<T, DIMENSIONS>::ZERO,
    };

    pub fn new(min: Vector<T, DIMENSIONS>, max: Vector<T, DIMENSIONS>) -> Self {
        Self { min, max }
    }

    pub fn contains_point(&self, point: Vector<T, DIMENSIONS>) -> bool {
        point.greater_than_per_component(self.min).all()
            && point.less_than_per_component(self.max).all()
    }

    pub fn new_with_min_corner_and_size(
        min_corner: Vector<T, DIMENSIONS>,
        size: Vector<T, DIMENSIONS>,
    ) -> Self {
        Self {
            min: min_corner,
            max: min_corner + size,
        }
    }

    /// Returns the area of a 2D `Rect`, or the volume of a 3D `Rect`
    pub fn area(&self) -> T {
        let size = self.max - self.min;
        let mut a = size[0];
        for v in &size.0[0][1..] {
            a = a * *v;
        }
        a
    }

    /// Creates a new `Rect` that encompasses `self` and `other`
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Creates a new `Rect` with only the part that is contained in both `Rect`s
    /// If they don't overlap this will be empty.
    pub fn intersection(self, other: Self) -> Self {
        Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }
}
