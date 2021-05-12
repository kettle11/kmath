use bencher::{benchmark_group, benchmark_main, Bencher};

fn mat4_mul_100k(b: &mut Bencher) {
    use kmath::*;

    let translation = Matrix::from_translation(Vector::new([1. as f32, 2., 3.]));
    let scale = Matrix::<f32, 4, 4>::from_scale(10.);

    let translation_scale = translation * scale;
    b.iter(|| {
        let mut v_out = Matrix::IDENTITY;
        for _ in 0..100_000 {
            v_out = v_out * translation_scale;
        }
        v_out
    })
}

fn mat4_mul_100k_glam(b: &mut Bencher) {
    use glam::*;
    let translation = Mat4::from_translation([1., 2., 3.].into());
    let scale = Mat4::from_scale(Vec3::splat(10.));

    let translation_scale = translation * scale;

    b.iter(|| {
        let mut v_out = Mat4::IDENTITY;
        for _ in 0..100_000 {
            v_out = v_out * translation_scale;
        }
        v_out
    })
}

benchmark_group!(benches, mat4_mul_100k, mat4_mul_100k_glam);
benchmark_main!(benches);
