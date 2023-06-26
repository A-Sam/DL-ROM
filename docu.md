# Target Scenarios
1. `2d_cylinder_CFD`
    - ` python main.py -N 100 -B 32 -d_set 2d_cylinder_CFD --train --simulate --test` validation_freq = 1
    - ` python main.py -N 100 -B 32 -d_set 2d_cylinder_CFD --train --transfer --simulate --test` validation_freq = 10
    - ` python main.py -N 100 -B 32 -d_set 2d_cylinder_CFD --train --transfer --simulate --test` validation_freq = 20
1. `airfoil80x320_data`
    - ` python main.py -N 100 -B 32 -d_set 2d_airfoil --train --simulate --test` validation_freq = 10
    - ` python main.py -N 100 -B 32 -d_set 2d_airfoil --train --transfer --simulate --test` validation_freq = 10
3. `sea_surface_noaa`
4. `sq_cyl_vort`
# Target settings

$$
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$