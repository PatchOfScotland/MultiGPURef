
let dot (c:(f32, f32)): f32 =
    c.0 * c.0 + c.1 * c.1

let ac (c1:(f32, f32)) (c2:(f32, f32)): (f32, f32) =
    (c1.0 + c2.0, c1.1 + c2.1)

let mc (c1:(f32, f32)) (c2:(f32, f32)): (f32, f32) =
    (c1.0 * c2.0 - c1.1 * c2.1, c1.0 * c2.1 + c1.1 * c2.0)

let divergence (c : (f32,f32)) (d:i32):i32 =
    let (_, done) =
        loop (z, i) = (c, 0) while i < d && dot(z) < 4.0f32 do
            (ac c (mc z z), i + 1)
    in done

let main  (n:i64) (m:i64)  (d:i32): [n][m]i32 =
    let css = tabulate_2d n m (\i j -> (((f32.i64 i) / (f32.i64 n))*4 - 2, ((f32.i64 j) / (f32.i64 m))*4 - 2))
    in
    map (\cs -> map (\c -> divergence c d) cs) css