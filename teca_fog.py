#!/home/obrienta/projects/teca-development/install/bin/python3 
from mpi4py import MPI
import teca
import numpy as np
from teca_python_vertical_reduction import teca_python_vertical_reduction
from calculate_cloud_base import calculate_cloud_base_sigma

class teca_fog(teca_python_vertical_reduction):
    """A TECA algorithm for calculating cloud base height and fog."""

    clw_var = "clw"
    ta_var = "ta"
    hus_var = "hus"
    ps_var = "ps"
    ts_var = "ts"
    ptop = 5000
    clw_threshold = 5e-5
    cloud_base_threshold = 400
    fill_value = -1e20

    def set_clw_var(self, clw_var):
        """ Sets the 3D cloud water content variable name in the input file"""
        self.clw_var = clw_var

    def set_ta_var(self, ta_var):
        """ Sets the 3D temperature variable name in the input file"""
        self.ta_var = ta_var

    def set_hus_var(self, hus_var):
        """ Sets the 3D specific humidity variable name in the input file"""
        self.hus_var = hus_var

    def set_ps_var(self, ps_var):
        """ Sets the surface pressure variable name in the input file"""
        self.ps_var = ps_var

    def set_ts_var(self, ts_var):
        """ Sets the surface temperature variable name in the input file"""
        self.ts_var = ts_var

    def set_ptop(self, ptop):
        """ Sets the model top pressure (in Pa)"""
        self.ptop = ptop

    def set_clw_threshold(self, clw_threshold):
        """ Sets the cloud water threshold for defining cloud presence [kg/kg]"""
        self.clw_threshold = clw_threshold

    def set_cloud_base_threshold(self, cloud_base_threshold):
        """ Sets the cloud base height threshold for defining fog [m]"""
        self.cloud_base_threshold = cloud_base_threshold

    def get_point_array_names(self):
        """ Returns the names of the output arrays """
        return ["fog", "z_cloud_base"]

    def set_fill_value(self, fill_value):
        """ Sets the value to use for indicating no-cloud presence in z_cloud_base"""
        self.fill_value = fill_value

    def get_fill_value(self):
        """ Gets the fill value used in z_cloud_base"""
        return self.fill_value

    def request(self, port, md_in, req_in):
        """ Define the TECA request phase for this algorithm"""

        self.set_dependent_variables([
            self.clw_var,
            self.ta_var,
            self.hus_var,
            self.ps_var,
            self.ts_var,
            ]
            )

        return super().request(port, md_in, req_in)

    def report(self, port, md_in):
        """ Define the TECA report phase for this algorithm"""
        md = teca.teca_metadata(md_in[0])

        fog_atts = teca.teca_array_attributes(
            teca.teca_double_array_code.get(),
            teca.teca_array_attributes.point_centering,
            0, 'fraction', 'Fog Frequency',"",
            self.fill_value,
        )

        cloud_base_atts = teca.teca_array_attributes(
            teca.teca_double_array_code.get(),
            teca.teca_array_attributes.point_centering,
            0, 'm', 'Cloud Base Height',"",
            self.fill_value,
        )

        # add the variables
        self.add_derived_variable_and_attributes("fog", fog_atts)
        self.add_derived_variable_and_attributes("z_cloud_base", cloud_base_atts)

        return super().report(port, md_in)

    def execute(self, port, data_in, req):
        """ Define the TECA execute phase for this algorithm.
            Outputs a TECA table row, which is intended to be used in conjunction
            with teca_table_reduce, and teca_table_sort.
        
        """

        # get the input mesh
        in_mesh = teca.as_const_teca_cartesian_mesh(data_in[0])

        # initialize the output mesh from the super function
        out_mesh = teca.as_const_teca_cartesian_mesh(super().execute(port, data_in, req))

        # get sigma
        sigma = in_mesh.get_z_coordinates()

        # get horizontal coordinates
        jx = in_mesh.get_x_coordinates()
        iy = in_mesh.get_y_coordinates()

        def reshape3d(in_var):
            return np.reshape(in_var, [len(sigma), len(iy), len(jx)])
        def reshape2d(in_var):
            return np.reshape(in_var, [len(iy), len(jx)])

        # get temperature, humidity, surface pressure, and surface temperature
        clw = reshape3d(in_mesh.get_point_arrays()[self.clw_var])
        ta = reshape3d(in_mesh.get_point_arrays()[self.ta_var])
        hus = reshape3d(in_mesh.get_point_arrays()[self.hus_var])
        ps = reshape2d(in_mesh.get_information_arrays()[self.ps_var])
        ts = reshape2d(in_mesh.get_information_arrays()[self.ts_var])

        # calculate cloud base height
        zbase = calculate_cloud_base_sigma(
            clw.astype(np.float64),
            ta.astype(np.float64),
            hus.astype(np.float64),
            ps.astype(np.float64),
            ts.astype(np.float64),
            np.array(sigma).astype(np.float64),
            np.float64(self.ptop),
            missing_value = np.float64(self.fill_value),
            clw_threshold = np.float64(self.clw_threshold),
        )

        # calculate fog
        fog = np.array(zbase <= self.cloud_base_threshold).astype(np.float64)

        out_arrays = out_mesh.get_point_arrays()
        out_arrays['fog'] = fog.ravel()
        out_arrays['z_cloud_base'] = zbase.ravel()


        # return the current table row
        return out_mesh


def construct_teca_pipeline(\
        files_regex,
        output_filename,
        be_verbose = True,
        x_axis_variable = "jx",
        y_axis_variable = "iy",
        z_axis_variable = "kz",
        clw_var = "clw",
        ta_var = "ta",
        hus_var = "hus",
        ps_var = "ps",
        ts_var = "ts",
        ptop = 5000,
        clw_threshold = 5e-5,
        cloud_base_threshold = 400,
        start_month_index = None,
        end_month_index = None,
        ):
    """Construct the TECA pipeline for this application.

    input:
    ------

        files_regex             : The regex string that will yield paths to input files

        output_filename         : The output filename template. The substring "%t%" will be substituted
                                  for a datestamp.

        be_verbose              : Flags whether to turn on TECA's verbose flags

        x_axis_variable         : The variable name of the x coordinate

        y_axis_variable         : The variable name of the y coordinate

        z_axis_variable         : The variable name of the z coordinate

        clw_var                 : The variable name for the cloud liquid water variable

        ta_var                  : The variable name for temperature

        hus_var                 : The variable name for specific humidity

        ps_var                  : The variable name for surface pressure

        ts_var                  : The variable name for surface temperature

        ptop                    : The model top pressure

        clw_threshold           : The minimum threshold in clw for determining cloud presence

        cloud_base_threshold    : The maximum cloud base height for determining fog

        start_month_index       : The index of the first month to process

        end_month_index         : The index of the last month to process

    """

    if "%t%" not in output_filename:
        raise RuntimeError(r"The output file name must have the time template %t% somewhere in the string")

    # initialize the pipeline stages
    pipeline_stages = []

    # reader
    cfr = teca.teca_cf_reader.New()
    cfr.set_x_axis_variable(x_axis_variable)
    cfr.set_y_axis_variable(y_axis_variable)
    cfr.set_z_axis_variable(z_axis_variable)
    cfr.set_files_regex(files_regex)
    pipeline_stages.append(cfr)

    # Normalize coordinates
    norm = teca.teca_normalize_coordinates.New()
    pipeline_stages.append(norm)

    # cloud base and fog
    fog = teca_fog.New()
    fog.set_clw_var(clw_var)
    fog.set_ta_var(ta_var)
    fog.set_hus_var(hus_var)
    fog.set_ps_var(ps_var)
    fog.set_ts_var(ts_var)
    fog.set_ptop(ptop)
    fog.set_clw_threshold(clw_threshold)
    fog.set_cloud_base_threshold(cloud_base_threshold)
    pipeline_stages.append(fog)

    # temporal reduction
    tre = teca.teca_temporal_reduction.New()
    tre.set_interval("monthly")
    tre.set_operator("average")
    tre.set_point_arrays(fog.get_point_array_names())
    tre.set_thread_pool_size(-1)
    tre.set_stream_size(2)
    tre.set_verbose(1)
    tre.set_verbose(int(be_verbose))
    pipeline_stages.append(tre)

    # executive
    exe = teca.teca_index_executive.New()
    exe.set_verbose(int(be_verbose))

    # writer
    tfw = teca.teca_cf_writer.New()
    tfw.set_file_name(output_filename)
    tfw.set_point_arrays(fog.get_point_array_names())
    tfw.set_thread_pool_size(1)
    tfw.set_executive(exe)
    tfw.set_steps_per_file(8)
    if start_month_index is not None:
        tfw.set_first_step(start_month_index)
    if end_month_index is not None:
        tfw.set_last_step(end_month_index)
    tfw.set_verbose(int(be_verbose))
    pipeline_stages.append(tfw)

    # connect the pipeline
    for n in range(1,len(pipeline_stages)):
        pipeline_stages[n].set_input_connection(\
                pipeline_stages[n-1].get_output_port())

    # return the last stage of the pipeline
    return pipeline_stages[-1]


if __name__ == "__main__":
    import argparse

    # construct the command line arguments
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--input_regex',
        help = "A regex that points to the files containing surface temperature",
        required = True)
    parser.add_argument('--output_file',
        help = "The name of the file to write to disk." ,
        required = True)
    parser.add_argument('--verbose',
        help = "Indicates whether to turn on verbose output.",
        action = 'store_true',
        default = False)
    parser.add_argument('--x_axis_variable',
        help = "The variable name of the x coordinate",
        default = "jx")
    parser.add_argument('--y_axis_variable',
        help = "The variable name of the y coordinate",
        default = "iy")
    parser.add_argument('--z_axis_variable',
        help = "The variable name of the z coordinate",
        default = "kz")
    parser.add_argument('--clw_var',
        help = "The variable name for the cloud liquid water variable",
        default = "clw")
    parser.add_argument('--ta_var',
        help = "The variable name for temperature",
        default = "ta")
    parser.add_argument('--hus_var',
        help = "The variable name for specific humidity",
        default = "hus")
    parser.add_argument('--ps_var',
        help = "The variable name for surface pressure",
        default = "ps")
    parser.add_argument('--ts_var',
        help = "The variable name for surface temperature",
        default = "ts")
    parser.add_argument('--ptop',
        help = "The model top pressure",
        default = 5000)
    parser.add_argument('--clw_threshold',
        help = "The minimum threshold in clw for determining cloud presence [kg/kg]",
        default = 5e-5)
    parser.add_argument('--cloud_base_threshold',
        help = "The maximum cloud base height for determining fog [m]",
        default = 400)
    parser.add_argument('--start_month_index',
        help = "The index of the first month to process",
        default = None)
    parser.add_argument('--end_month_index',
        help = "The index of the last month to process",
        default = None)


    # parse the command line arguments
    args = parser.parse_args()

    # construct the TECA pipeline
    pipeline = construct_teca_pipeline(
        files_regex = args.input_regex,
        output_filename = args.output_file,
        be_verbose = args.verbose,
        x_axis_variable = args.x_axis_variable,
        y_axis_variable = args.y_axis_variable,
        z_axis_variable = args.z_axis_variable,
        clw_var = args.clw_var,
        ta_var = args.ta_var,
        hus_var = args.hus_var,
        ps_var = args.ps_var,
        ts_var = args.ts_var,
        ptop = args.ptop,
        clw_threshold = float(args.clw_threshold),
        cloud_base_threshold = float(args.cloud_base_threshold),
        start_month_index = int(args.start_month_index),
        end_month_index = int(args.end_month_index),
    )

    # run the pipeline
    pipeline.update()

