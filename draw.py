import numpy as np
import meshio
import math

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error


def read_vector_from_file(filename):
    try:
        vector = np.loadtxt(filename)
        return vector
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
    
def read_mesh(filename):
    stick = meshio.read(filename)
    nodes = stick.points
    cells = stick.cells_dict['tetra']
    print("load mesh from %s: %d nodes, %d cells"%(filename,len(nodes),len(cells)))
    return nodes,cells


def compute_metrics(cell_centers,true_values, predicted_values, colorbar_min, colorbar_max):
    # MSE
    mse = mean_squared_error(true_values, predicted_values)
    
    # RMSE
    rmse = math.sqrt(mse)
    
    # SSIM
    ssim_value = ssim(true_values, predicted_values, data_range = colorbar_max - colorbar_min, win_size = 401)
    # PSNR
    psnr_value = psnr((true_values - colorbar_min)/(colorbar_max - colorbar_min), 
                      (predicted_values - colorbar_min)/(colorbar_max - colorbar_min), data_range=1)
    
    # normalized similarity
    normalized_corr = np.corrcoef(true_values.flatten().reshape(1, -1), predicted_values.flatten().reshape(1, -1))[0, 1]
    return mse, rmse, psnr_value,ssim_value, normalized_corr



def draw_multiple_we_slices(nodes, cells, we_list, L, Lz, path1, path2, path3,num_slices=10, colorbar_min=None, colorbar_max=None, cmap = "coolwarm"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    cell_centers = np.array([nodes[cell].mean(axis=0) for cell in cells])

    z_slices = np.linspace(0.05 * Lz , 0.95 * Lz, num_slices)[::-1]  # 逆序排列

    num_we = len(we_list)
    fig, axes = plt.subplots(num_slices, num_we, figsize=(3 *0.85* num_we, 3 *0.6* num_slices), constrained_layout=False)

    x_min, x_max = -L / 2, L / 2
    y_min, y_max = -L / 2, L / 2

    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(xi, yi)

    vmin = colorbar_min if colorbar_min is not None else min([np.min(we) for we in we_list])
    vmax = colorbar_max if colorbar_max is not None else max([np.max(we) for we in we_list])

    # ground truth
    true_we = we_list[0]

    metrics_summary = []
    for col, we in enumerate(we_list):
        points = np.column_stack((cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]))
        values = we

        for row, z_val in enumerate(z_slices):
            Z = np.full(X.shape, z_val)
            
            grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            we_grid = griddata(points, values, grid_points, method='linear').reshape(X.shape)

            im = axes[row, col].imshow(we_grid, extent=(x_min, x_max, y_min, y_max),
                                       origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_xlabel("")
            axes[row, col].set_ylabel("")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])


        # compute metrics
        if col > 0:  # skip ground truth
            mse, rmse, psnr_val,ssim_val, cosine_sim = compute_metrics(cell_centers,true_we, we, colorbar_min, colorbar_max)
            metrics_summary.append(
                f"Comparing WE 0 with WE {col}:\n"
                f"RMSE = {rmse/1e6:.6f} MPa, SSIM = {ssim_val:.6f}, PSNR = {psnr_val:.6f}, Cosine Similarity = {cosine_sim:.6f}\n"
                f"{'-' * 40}\n"
            )
    plt.subplots_adjust(left=0.08, right=0.83, top=0.95, bottom=0.05, hspace=0.05, wspace=0.07)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label("Hardness (we)")
    


    os.makedirs(os.path.dirname(path1), exist_ok=True)
    fig.savefig(path1, dpi=300, bbox_inches='tight')

    plt.show()

    # save to the path
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2, 'w') as f:
        f.writelines(metrics_summary)

    for summary in metrics_summary:
        print(summary)

    print(f"Figure saved to: {path1}")
    print(f"Metrics saved to: {path2}")


def draw(path, L = 0.01, colorbar_max= 4e6, dz = 0, cmap = "coolwarm"):

    nodes,cells=read_mesh("data/"+path + ".msh")
    nodes[:,2] += dz
    we = []
    we.append(read_vector_from_file("data/" + path +"-we-forward.txt"))

    path_string = "results/" + path + "/"
    path1 = "results/pictures/" + path  +".png"
    path2 = "results/indices/" + path  +".txt"
    path3 = "results/pictures/" + path + "-similarity.png"
    we.append(read_vector_from_file(path_string + "we.txt"))# "we-ref.txt" for reference
    

    draw_multiple_we_slices(nodes, cells, we, 2*L, L, path1, path2,path3, num_slices=8,colorbar_min=0.001e6, colorbar_max= colorbar_max, cmap = cmap)

#draw("box-torus")
#draw("box-cone-soft-1",colorbar_max=0.15e6)
#draw("hyperboloid",L=0.1, colorbar_max=2e6, dz = 0.05)
draw("box-3",colorbar_max=1.5e6, cmap = "jet")