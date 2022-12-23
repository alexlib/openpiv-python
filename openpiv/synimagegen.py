from typing import List, Tuple, Optional
import pathlib
import numpy as np
import scipy
from scipy.special import erf
import scipy.interpolate
import matplotlib.pyplot as pl

"""This module generates synthetic images for OpenPIV """

__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANy WARRANTy; without even the implied warranty of
MERCHANTABILITy or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

you should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


class ContinousFlowField:
    """Continuous flow field class"""

    def __init__(self, data, inter=False):
        """
        Checks if the continous flow should be created from a set of data points
        if so it interpolates them for a continuous flow field
        """
        self.inter = inter
        self.data = data

    # Defining a synthetic flow field
    def f_u(self, x, y):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        # example for synthetic U velocity
        u = 2.5 + 0.5 * np.sin((x**2 + y**2) / 0.01)
        return u

    def f_v(self, x, y):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        # example for synthetic V velocity
        v = 0.5 + 0.1 * np.cos((x**2 + y**2) / 0.01)
        return v

    def get_uv(self, x, y):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        # return the U and V velocity at a certain position
        if self.inter:
            f_u = scipy.interpolate.interp2d(self.data[:, 0], self.data[:, 1], self.data[:, 2])
            f_v = scipy.interpolate.interp2d(self.data[:, 0], self.data[:, 1], self.data[:, 3])
            return f_u(x, y)[0], f_v(x, y)[0]
        else:
            return self.f_u(x, y), self.f_v(x, y)

    def create_syn_quiver(self, number_of_grid_points, path=None):
        """return and save a synthetic flow map

        Args:
            number_of_grid_points (_type_): _description_
            path (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        x, y = np.meshgrid(
            np.linspace(0, 1, number_of_grid_points),
            np.linspace(0, 1, number_of_grid_points),
        )
        U = np.zeros(x.shape)
        V = np.zeros(y.shape)
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                u, v = self.get_uv(x[r, c], y[r, c])
                U[r, c] = u
                V[r, c] = v

        m = np.sqrt(np.power(U, 2) + np.power(V, 2))
        fig = pl.quiver(
            x,
            y,
            U,
            V,
            m,
            clim=[1.5, m.max()],
            scale=100,
            width=0.002,
            headwidth=6,
            minshaft=2,
        )
        cb = pl.colorbar(fig)
        cb.set_clim(vmin=1.5, vmax=m.max())

        if not path:
            pl.savefig("syn_quiver.png", dpi=400)
            pl.close()
        else:
            pl.savefig(path + "syn_quiver.png", dpi=400)
            pl.close()

        return x, y, U, V


def create_synimage_parameters(
    input_data: np.array = None,
    x_bound: Tuple = (0, 1),
    y_bound: Tuple = (0, 1),
    image_size: Tuple[int, int] = (256, 256),
    path: Optional[pathlib.Path] = None,
    inter: bool = False,
    den: float = 0.008,
    per_loss_pairs: int = 2,
    par_diam_mean: float = 3.87, 
    par_diam_std: float = 1.5,
    par_int_std: float = 0.25,
    dt: float = 0.1,
):
    """Creates the synthetic image with the synthetic image parameters

    Parameters
    ----------
    input_data: None or numpy array
        If you have data from which to genrate the flow feild the synthetic image.
        It should be passed on as a numpy array with columns being (x grid position,y grid position,U velocity at (x,y) grid point,V velocity at (x,y) grid point)
        Else, pass None and define a synthetic flow field in ContinousFlowField class.

    x_bound,y_bound: list/tuple of floats
        The boundries of interest in the synthetic flow field.

    image_size: list/tuple of ints
        The desired image size in pixels.

    path: str('None' for no generating data)
        Path to txt file of input data.

    inter: boolean
        False if no interpolation of input data is needed.
        True if there is data you want to interpolate from.

    den: float
        Defines the number of particles per image.

    per_loss_pairs: float
        Percentage of synthetic pairs loss.

    par_diam_mean: float
        Mean particle diamter in pixels.

    par_diam_std: float
        Standard deviation of particles diamter in pixels.

    par_int_std: float
        Standard deviation of particles intensities.

    dt: float
        Synthetic time difference between both images.

    Returns
    -------
    ground_truth: ContinousFlowField class
        The synthetic ground truth as a ContinousFlowField class.

    cv:
        Convertion value to convert U,V from pixels/images to meters/seconds.

    x_1,y_1: numpy array
        Position of particles in the first synthetic image.

    u_par,v_par: numpy array
        Velocity speeds for each particle.

    par_diam1: numpy array
        Particle diamters for the first synthetic image.

    par_int1: numpy array
        Particle intensities for the first synthetic image.

    x_2,y_2: numpy array
        Position of particles in the second synthetic image.

    par_diam2: numpy array
        Particle diamters for the second synthetic image.

    par_int2: numpy array
        Particle intensities for the second synthetic image.
    """

    # Data processing

    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()

        data = [line.split("\t") for line in data]
        data = np.array(data).astype(float)
        data = np.array(
            [
                line
                for line in data.tolist()
                if 1.2 * x_bound[1] >= line[1] >= 0.8 * x_bound[0]
                and 1.2 * y_bound[1] >= line[2] >= 0.8 * y_bound[0]
            ]
        )

    else:
        data = input_data

    if inter:
        cff = ContinousFlowField(data, inter=inter)
    else:
        cff = ContinousFlowField(None)

    # Creating syn particles

    num_of_par = int(image_size[0] * image_size[1] * den)
    num_of_lost_pairs = num_of_par * (per_loss_pairs / 100)
    x_1 = np.random.uniform(x_bound[0] * 0.8, x_bound[1] * 1.2, num_of_par)
    y_1 = np.random.uniform(y_bound[0] * 0.8, y_bound[1] * 1.2, num_of_par)
    par_diam1 = np.random.normal(par_diam_mean, par_diam_std, num_of_par)
    particle_centers = np.random.uniform(size=num_of_par) - 0.5
    par_int1 = np.exp(-(particle_centers**2) / (2 * par_int_std**2))
    u_par = np.zeros(x_1.shape)
    v_par = np.zeros(y_1.shape)
    x_2 = np.zeros(x_1.shape)
    y_2 = np.zeros(y_1.shape)
    par_diam2 = np.zeros(par_diam1.shape)
    par_int2 = np.zeros(par_int1.shape)

    def move_particle(i):
        u_par[i], v_par[i] = cff.get_uv(x_1[i], y_1[i])
        x_2[i] = x_1[i] + u_par[i] * dt
        y_2[i] = y_1[i] + v_par[i] * dt
        par_diam2[i] = par_diam1[i]
        par_int2[i] = par_int1[i]

    cpl = 0
    for i in range(num_of_par):
        if cpl < num_of_lost_pairs:
            if -0.4 > particle_centers[i] or 0.4 < particle_centers[i]:
                per_to_lose = 1 - (0.5 - np.abs(particle_centers[i])) / 0.1
                if np.random.uniform() < min(per_loss_pairs / 10, 1) * per_to_lose:
                    x_2[i] = np.random.uniform(x_bound[0] * 0.8, x_bound[1] * 1.2)
                    y_2[i] = np.random.uniform(y_bound[0] * 0.8, y_bound[1] * 1.2)
                    par_diam2[i] = np.random.normal(par_diam_mean, par_diam_std)
                    par_int2[i] = np.exp(
                        -((np.random.uniform() - 0.5) ** 2) / (2 * par_int_std**2)
                    )
                    cpl += 1
                else:
                    move_particle(i)
            else:
                move_particle(i)
        else:
            move_particle(i)

    print(
        "Requested pair loss:",
        str(int(num_of_lost_pairs)),
        " Actual pair loss:",
        str(cpl),
    )
    xy_1 = np.transpose(np.vstack((x_1, y_1, u_par, v_par, par_diam1, par_int1)))
    xy_2 = np.transpose(np.vstack((x_2, y_2, par_diam2, par_int2)))

    # Choosing particles in boundary area

    bounded_xy_1 = np.asarray(
        [
            xy
            for xy in xy_1
            if x_bound[1] >= xy[0] >= x_bound[0] and y_bound[1] >= xy[1] >= y_bound[0]
        ]
    )
    bounded_xy_2 = np.asarray(
        [
            xy
            for xy in xy_2
            if x_bound[1] >= xy[0] >= x_bound[0] and y_bound[1] >= xy[1] >= y_bound[0]
        ]
    )

    # Tranforming coordinates into pixels

    x1 = ((bounded_xy_1[:, 0] - x_bound[0]) / (x_bound[1] - x_bound[0])) * image_size[0]
    y1 = ((bounded_xy_1[:, 1] - y_bound[0]) / (y_bound[1] - y_bound[0])) * image_size[1]

    x2 = ((bounded_xy_2[:, 0] - x_bound[0]) / (x_bound[1] - x_bound[0])) * image_size[0]
    y2 = ((bounded_xy_2[:, 1] - y_bound[0]) / (y_bound[1] - y_bound[0])) * image_size[1]

    conversion_value = (
        min(
            (x_bound[1] - x_bound[0]) / image_size[0],
            (y_bound[1] - y_bound[0]) / image_size[1],
        )
        / dt
    )

    return (
        cff,
        conversion_value,
        x1,
        y1,
        bounded_xy_1[:, 2],
        bounded_xy_1[:, 3],
        bounded_xy_1[:, 4],
        bounded_xy_1[:, 5],
        x2,
        y2,
        bounded_xy_2[:, 2],
        bounded_xy_2[:, 3],
    )


def generate_particle_image(
    height: int,
    width: int,
    x: np.array,
    y: np.array,
    particle_diameters: np.array,
    particle_max_intensity: float,
    bit_depth: int=8,
):
    """Creates the synthetic image with the synthetic image parameters
    Should be run with the parameters of each image (first,second) separately.

    Parameters
    ----------
    height, width: int
        The number of pixels in the desired output image.

    x,y: numpy array
        The x and y positions of the particles, created by create_synimage_parameters().

    particle_diameters, particle_max_intensity: numpy array
                The intensities and diameters of the particles, created by create_synimage_parameters().

        bit_depth: int
                The bit depth of the desired output image.

    Returns
    -------
    Image: numpy array
        The desired synthetic image.

    """
    render_fraction = 0.75
    sqrt8 = np.sqrt(8)

    image_out = np.zeros([height, width])

    minRenderedCols = (x - render_fraction * particle_diameters).astype(int)
    maxRenderedCols = (np.ceil(x + render_fraction * particle_diameters)).astype(int)
    minRenderedRows = (y - render_fraction * particle_diameters).astype(int)
    maxRenderedRows = (np.ceil(y + render_fraction * particle_diameters)).astype(int)

    index_to_render = []

    for i in range(x.size):
        if (
            1 < minRenderedCols[i]
            and maxRenderedCols[i] < width
            and 1 < minRenderedRows[i]
            and maxRenderedRows[i] < height
        ):
            index_to_render.append(i)

    for ind in index_to_render:
        max_int = particle_max_intensity[ind]
        par_diam = particle_diameters[ind]

        bl = max(minRenderedCols[ind], 0)
        br = min(maxRenderedCols[ind], width)
        bu = max(minRenderedRows[ind], 0)
        bd = min(maxRenderedRows[ind], height)

        for c in range(bl, br):
            for r in range(bu, bd):
                image_out[r, c] = image_out[r,c] + max_int * par_diam**2 * np.pi / 32 * \
                (
                    erf(sqrt8 * (c - x[ind] - 0.5) / par_diam)
                    - erf(sqrt8 * (c - x[ind] + 0.5) / par_diam)
                ) * (
                    erf(sqrt8 * (r - y[ind] - 0.5) / par_diam)
                    - erf(sqrt8 * (r - y[ind] + 0.5) / par_diam)
                )

    noise_mean = 2 ** (bit_depth * 0.3)
    noise_std = 0.25 * noise_mean
    noise = noise_std * np.random.randn(height, width) + noise_mean
    return (image_out * (2**bit_depth * 2.8**2 / 8) + noise).astype(int)[::-1]
