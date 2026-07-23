# Commands for reproducing figures

| Figure | Command (run in the enclosing directory) |
|--------|------------------------------------------|
| 1B | `python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv -o paper/figures/1b.pdf` |
| 1C | `python paper/response.py --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-space 100 100 --N-ori 12 --N-osi 7 -m eigvals --eps 1e-2 -o paper/figures/1c.pdf` |
| 1D (left) | `python paper/response.py --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-space 100 100 --N-ori 12 --N-osi 7 -m compare -k space_ori --dh 10000 -o paper/figures/1d_a.pdf` |
| 1D (right) | `python paper/response.py --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-space 100 100 --N-ori 12 --N-osi 7 -m compare -k ori_osi --dh 10000 --tau-i 1.0 -o paper/figures/1d_b.pdf` |
| 2A | `python paper/contourplot_space.py E -o paper/figures/2a.pdf` |
| 2B | `python paper/contourplot_space.py E --rho 1 2 -o paper/figures/2b.pdf` |
| 2C | `python paper/response.py --wee 4.5 1.5 1.5 --wei 3 3 2 --wie 2 2.5 2.5 --wii 0 4 4 --N-space 100 100 -m space --N-ori 12 --N-osi 7 --normalize --dh 10000 --tau-i 0.25 --maxiter 1000 -o paper/figures/2c.pdf` |
| 2D | `python paper/contourplot_space.py r0 -y 0 15 --rho 0.72 --w00 5 -l -1 1 -n 9 --shade 0.443 0.691 -o paper/figures/2d.pdf` |
| 2E | `python paper/contourplot_space.py rmin -y 0 15 --rho 0.72 --w00 5 -s linear -l 0 2 -n 9 --shade 0.260 0.530 -o paper/figures/2e.pdf` |
| 2F | `python paper/contourplot_space.py dr1dw11 -y 0 15 --rho 0.72 --w00 5 -s halflog -l -1.25 2 -n 14 -t 0.5 -o paper/figures/2f.pdf` |
| 2G | `python paper/contourplot_space.py decay -x -5 2.5 --rho 0.72 -o paper/figures/2g.pdf` |
| 3A | `python paper/contourplot_space.py EI -y 0 15 --rho 0.72 -o paper/figures/3a.pdf` |
| 3B | `python paper/contourplot_space.py rEI -y 0 15 --rho 0.72 --w00 5 -l -0.5 1.25 -s halflog -n 8 -t 2 -o paper/figures/3b.pdf` |
| 4A | `python paper/response.py --wee 1 --wei 4 --wie 4 --wii 0 --kee 0.5 0.5 --kei 0.25 0.5 --kie 0.25 0.5 --kii 0 0 --N-space 100 100 --N-ori 12 --N-osi 7 -m ori --dh 10000 -o paper/figures/4a.pdf` |
| 4B | `python paper/contourplot_ori.py -o paper/figures/4b.pdf` |
| 4C | `python paper/response.py --wee 1.5 1.5 --wei 3 3 --wie 3 3 --wii 5 5 --kee 0.15 0.15 --kei 0.5 0.3 --kie 0.4 0.15 --kii 0.5 0.5 --N-space 100 100 --N-ori 12 --N-osi 7 -m space_ori --dh 10000 --normalize --rlim 35 300 -o paper/figures/4c.pdf` |
| 4D | `python paper/contourplot_space.py E -x -2 2 -y -2 2 --rho 0.72 --w00 0.2 --ori -o paper/figures/4d.pdf` |
| 5A | `paper/mean_response_gain.py 2 1 3 --N-space 100 100 --N-ori 12 --N-osi 7 --scale 10000 --rtol 5e-5 -o paper/figures/5a.pdf` |
| 5B | Same as Figure 2B
| 5C | `python paper/contourplot_space.py dr0dg -y 0 15 --rho 0.72 --w00 5 -l -0.25 1.25 -n 7 -s neghalflog -t 2 -o paper/figures/5c.pdf` |
| 5D | Same as Figure 4B |
| 6 | See `paper/model_fits/no_disorder/README.md` |
| S1 | `for RHO in 0.8 1 1.25; do for WEE in 1 2.5 5; do python paper/contourplot_space.py r0 -y 0 15 --rho $RHO --w00 $WEE -l -1.75 1 -n 12 --shade 0.443 0.691 -N 200 -o paper/figures/supp/S1/rho$RHO-wee$WEE.pdf; done; done` |
| S2 | `for RHO in 0.8 1 1.25; do for WEE in 1 2.5 5; do python paper/contourplot_space.py rmin -y 0 15 --rho $RHO --w00 $WEE -s linear -l 0 2 -n 9 --shade 0.260 0.530 -N 200 -o paper/figures/supp/S2/rho$RHO-wee$WEE.pdf; done; done` |
| S3 | `for RHO in 0.8 1 1.25; do for WEE in 1 2.5 5; do python paper/contourplot_space.py dr1dw11 -y 0 15 --rho $RHO --w00 $WEE -s halflog -l -4 2 -n 25 -t 0.001 -N 200 -o paper/figures/supp/S3/rho$RHO-wee$WEE.pdf; done; done` |
| S4 | `for RHO in 0.8 1 1.25; do for WEE in 1 2.5 5; do python paper/contourplot_space.py rEI -y 0 15 --rho $RHO --w00 $WEE -l -1 1 -s symlog -n 5 -t 0.2 -N 200 -o paper/figures/supp/S4/rho$RHO-wee$WEE.pdf; done; done` |
| S5 | `for RHO in 0.8 1 1.25; do for WEE in 1 2.5 5; do python paper/contourplot_space.py dr0dg -y 0 15 --rho $RHO --w00 $WEE -l -1 1 -n 11 -s symlog -N 200 -o paper/figures/supp/S5/rho$RHO-wee$WEE.pdf; done; done` |
| S6 | See `paper/model_fits/no_disorder/README.md` |
| S7A | `python paper/fit_kernel.py paper/rossi_data/bootstrap/EE.csv paper/rossi_data/bootstrap/EI.csv -d 0 -M 300 -l G -x 0 500 5 --ylabel prob --y-intercept 0.1 0.275 -o paper/figures/supp/S7_a.pdf` |
| S7B inset | `python paper/fit_rossi_ori.py paper/rossi_data/fig2h.csv -o paper/figures/supp/S7_b.pdf` |
| S7C inset | `python paper/plot_znamenskiy.py paper/znamenskiy_data/EI.csv -o paper/figures/supp/S7_c.pdf` |
| S7D inset | `python paper/plot_znamenskiy.py paper/znamenskiy_data/ori/EI.csv --ori -o paper/figures/supp/S7_d.pdf` |
| S7B-J | See `paper/model_fits/disordered/README.md` |
| S8A (left) | `python paper/plot_gain.py -b 0.5 -s 968 -y 1.55 -o paper/figures/supp/S8_a.pdf` |
| S8B (left) | `python paper/plot_gain.py -b 0.5 -k 0.5 -y 1.55 -o paper/figures/supp/S8_b.pdf` |
| S8C (left) | `python paper/plot_gain.py -b 0.5 -s 968 -k 0.5 -y 1.55 -o paper/figures/supp/S8_c.pdf` |
| S8A-C (center, right), S8D | See `paper/model_fits/no_disorder/README.md` |
| Psi-formula origin perturbation comparison | `python paper/plot_origin_perturbation_celltype_comparison.py --N-space 48 48 --heatmap-extent 100 100 --N-ori 8 --space-extent 400 --fit-index 0 --experiment-name origin_horizontal_perturbation --dh-values -10000 -5000 0 5000 10000 --seed 0 --max-neurons 50000 --skip-psi` |
| Psi-formula origin perturbation comparison, polar psi profiles | `python paper/plot_origin_perturbation_polar_celltype_comparison.py --N-space 48 48 --N-ori 8 --space-extent 400 --fit-index 0 --experiment-name origin_horizontal_perturbation_polar_psi --seed 0 --max-neurons 50000 --only-psi` |

# Additional commands
- The data files `paper/rossi_data/EI.csv` and `paper/rossi_data/bootstrap/EI.csv` are generated by running\
`python paper/process_rossi.py paper/rossi_data -o paper/rossi_data` and\
`python paper/process_rossi.py paper/rossi_data -b -o paper/rossi_data/bootstrap`\
respectively.

- The data files `paper/znamenskiy_data/EI.csv` and `paper/znamenskiy_data/IE.csv` are generated by running\
`python paper/process_znamenskiy.py paper/znamenskiy_data -o paper/znamenskiy_data`.

- The data files `paper/product_data/EI.csv` and `paper/product_data/IE.csv` used in the generation of Figure 1B are generated by running\
`python paper/get_product.py paper/rossi_data/EI.csv paper/znamenskiy_data/EI.csv -o paper/product_data/EI.csv` and\
`python paper/get_product.py paper/rossi_data/EI.csv paper/znamenskiy_data/IE.csv -o paper/product_data/IE.csv`\
respectively.

- To get the best-fit values and uncertainties of $\sigma_\mathrm{E} = 150 \pm 11$ and $\sigma_\mathrm{I} = 108 \pm 8$ as well as their geometric mean $\sqrt{\sigma_\mathrm{E}\sigma_\mathrm{I}} = 127.1 \pm 13.8$, run\
`python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv`

- To get the best-fit value of $\rho = 0.72$, run\
`python paper/uncertainty_rho.py 150.1898239023003 11.31920466 107.5719331775833 8.43638973`

- To get the 95% confidenence interval $[0.443, 0.691]$ in Figure 2D and S1 (these values are used in the commands in producing those figures, as seen from the above table of commands), run\
`python paper/uncertainty.py paper/chettih_data/r0_samples.csv 127.1 13.8`\
where the file `paper/chettih_data/r0_samples.csv` is generated by running\
`python paper/estimator.py paper/chettih_data/data.csv -e r0 --rmax 100 -m 1 -o paper/chettih_data/r0_samples.csv`\
and the file `paper/chettih_data/data.csv` is generated by running\
`python paper/process_chettih.py paper/chettih_data/bias.csv --format csv -o paper/chettih_data/data.csv`\
Note that the values for the 95% confidence interval differs slightly from run to run since these values are estimated from bootstrapping and thus there is randomness involved.

- Similarly, to get the 95% confidenence interval $[0.260, 0.530]$ in Figure 2E and S2 (these values are used in the commands in producing those figures, as seen from the above table of commands), run\
`python paper/uncertainty.py paper/chettih_data/r0_samples.csv paper/chettih_data/rmin_samples.csv 127.1 13.8 -e='f[1]-f[0]'`\
where the file `paper/chettih_data/rmin_samples.csv` is generated by running\
`python paper/estimator.py paper/chettih_data/data.csv -e rmin --rmax 300 -o paper/chettih_data/rmin_samples.csv`\
and the file `paper/chettih_data/data.csv` is generated by running\
`python paper/process_chettih.py paper/chettih_data/bias.csv --format csv -o paper/chettih_data/data.csv`\
Again, the values for the 95% confidence interval differs slightly from run to run since these values are estimated from bootstrapping and thus there is randomness involved.

- To get the best-fit value and uncertainty of $\kappa_\mathrm{EE} = 0.198 \pm 0.054$ used in constraining fitted model parameters in Figure 6, run\
`python paper/fit_rossi_ori.py paper/rossi_data/fig2h.csv`

- The values of pmax used in defining the connection probability in Figure S6 is computed with the command\
`python paper/calc_pmax.py`

# Raw data files
| Filename(s) | Source | Acquisition method |
|-------------|--------|--------------------|
| `paper/chettih_data/bias.csv` | Chettih and Harvey (2019) Figure 2G | Extracted from Matlab figure files provided by Chettih and Harvey |
| `paper/chettih_data/orientation.csv` | Chettih and Harvey (2019) Figure 3I | Same as above |
| `paper/rossi_data/fig1l_*.csv` | Rossi et al. (2020) Figure 1L | Obtained from their publicly available source code at https://doi.org/10.1038/s41586-020-2894-4, with slight modifications such that `opts.rmax` in `Code/Scripts/Figure_01.m` is set to `487.5` rather than `500`, and Gaussian smoothing is disabled by removing the line `rza = imgaussfilt(rza, sigma)` from `Code/Utils/rxyz_projection.m`. |
| `paper/rossi_data/fig2h.csv` | Rossi et al. (2020) Figure 2H | Extracted from Matlab figure file generated with their publicly available source code |
| `paper/znamenskiy_data/IPSP.csv` | Znamenskiy et al. (2024) Figure S4M | Extracted from Matlab figure file generated with their publicly available source code at https://doi.org/10.25418/crick.23295188 |
| `paper/znamenskiy_data/EPSP.csv` | Znamenskiy et al. (2024) Figure S5M | Same as above |
| `paper/znamenskiy_data/ori/IPSP.csv` | Znamenskiy et al. (2024) Figure S4L | Same as above |
| `paper/znamenskiy_data/ori/EPSP.csv` | Znamenskiy et al. (2024) Figure S5L | Same as above |
