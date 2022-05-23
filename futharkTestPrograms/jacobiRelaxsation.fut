let main [n] (xs: [n][n]f32) =
    loop ys = xs for _iter < 1000 do
        tabulate_2d n n (\i j ->
            let top = if j == 0 then 0 else ys[j-1,i]
            let bot = if j == n - 1 then 0 else ys[j+1,i]
            let lef = if i == 0 then 0 else ys[j,i-1]
            let rig = if i == n - 1 then 0 else ys[j,i+1]
            in (top + bot + lef + rig) / 4 
        )