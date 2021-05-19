use std::convert::TryInto;

use crate::*;

#[test]
fn to_translation_rotation_scale() {
    let base_m = [
        -16.864699217806663f32,
        5.3354828517975426,
        -98.42314803393586,
        0.,
        -77.429395652955307,
        -62.506756090694715,
        9.8789742299747747,
        0.,
        -60.994021448907333,
        77.874502053626955,
        14.672807413371247,
        0.,
        -150.59115600585938,
        156.88896179199219,
        416.95681762695312,
        1.,
    ];
    let m: Matrix<f32, 4, 4> = (&base_m).try_into().unwrap();

    println!("DECOMPOSED: {:#?}", m.to_translation_rotation_scale());

    let m: glam::Mat4 = glam::Mat4::from_cols_array(&base_m);
    println!("DECOMPOSED: {:#?}", m.to_scale_rotation_translation());
}
